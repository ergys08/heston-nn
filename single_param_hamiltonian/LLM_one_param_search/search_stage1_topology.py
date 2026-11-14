"""Stage 1 (v2): structured pair-block topology search on the 8-state subset.

Architecture = 8 single-qubit Ry, an ordered sequence of so(4) pair blocks
(six odd-Y generators: Y_i, Y_j, Y_iX_j, Y_iZ_j, X_iY_j, Z_iY_j on qubits
(i,j)), and 8 trailing Ry.  Each block costs 2 CNOTs after Qiskit
consolidation.  Fidelity only rises once blocks form a chain from the most
significant qubit (see discard_log.md), so candidate families are built
around the chain, applied in MPS order (LSB pair first):

  A  chain prefixes  (k-1,k),...,(1,2),(0,1), k=7..3  (blocks nearest the MSB kept)
  B  full chain minus one block
  C  full chain + ONE extra block of any pair at any position
  D  chain + partial second sweep (towards the MPS two-layer limit)
  E  chain with one nearest-neighbour block replaced by a long-range pair

Regimes: fixed (0 dof), rank1, rank2 (shared-direction models) and allvar
(every coefficient free; MPS-like bound).  Fits are warm-started from the
iterative-MPS solution (per state for allvar, of the mean state otherwise)
plus random restarts.
"""
from __future__ import annotations

import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ansatz import fit, fit_lowrank
from paulis import odd_y_strings
from states import load_states
from compile import cx_counts
from mps_init import mps_angles

ORDER = ["a", "b", "YX", "YZ", "XY", "ZY"]
CHAIN = [(6 - j, 7 - j) for j in range(7)]          # MPS order: (6,7),(5,6),...,(0,1)
SINGLES = [l for q in range(8) for l in odd_y_strings((q,))]


def block_labels(i, j):
    singles = {"a": odd_y_strings((i,)), "b": odd_y_strings((j,))}
    two = {l[i] + l[j]: l for l in odd_y_strings((i, j))}
    out = []
    for tok in ORDER:
        out += singles[tok] if tok in singles else [two[tok]]
    return out


def labels_for(blocks):
    """labels and parallel 'tokens' identifying each position for warm starts."""
    labels = list(SINGLES) + sum([block_labels(i, j) for i, j in blocks], []) + list(SINGLES)
    return labels


def tokens_for(blocks):
    toks = [("pre", q) for q in range(8)]
    seen = {}
    for b in blocks:
        n = seen.get(b, 0); seen[b] = n + 1
        toks += [(b, n, t) for t in ORDER]
    toks += [("post", q) for q in range(8)]
    return toks


def mps_theta_for(blocks, target, n_layers):
    """Warm start: MPS angles for the chain(s) mapped onto `blocks`; extra blocks 0."""
    lab, th, _ = mps_angles(target, n_layers)
    # MPS-order tokens: pre singles, layer blocks, Y0 after each layer
    mps_tok = [("pre", q) for q in range(8)]
    vals = {}
    k = 8
    seen = {}
    for L in range(n_layers):
        for b in CHAIN:
            n = seen.get(b, 0); seen[b] = n + 1
            for t in ORDER:
                vals[(b, n, t)] = th[k]; k += 1
        vals[("post", 0)] = vals.get(("post", 0), 0.0) + th[k]; k += 1   # final Y0 of each layer (approximate for L>1)
    for q in range(8):
        vals[("pre", q)] = th[q]
    return np.array([vals.get(t, 0.0) for t in tokens_for(blocks)])


def evaluate(labels, blocks, targets, regime, restarts=6, seed=0):
    n_layers = 2 if len(blocks) > 7 else 1
    mean = targets.mean(0); mean /= np.linalg.norm(mean)
    if regime == "allvar":
        init = np.stack([mps_theta_for(blocks, t, n_layers) for t in targets])
        r = fit(labels, targets, variable=list(range(len(labels))), init_var=init, restarts=restarts, seed=seed, maxiter=2000, init_scale=0.5, perturb=0.2)
        return r.mean_fidelity, r.min_fidelity
    if regime == "fixed":
        init = mps_theta_for(blocks, mean, n_layers)
        r = fit(labels, targets, variable=[], init_fixed=init, restarts=restarts, seed=seed, maxiter=2000, init_scale=0.5, perturb=0.2)
        return r.mean_fidelity, r.min_fidelity
    rank = int(regime[4:])
    init = mps_theta_for(blocks, mean, n_layers)
    best = None
    for s in range(restarts):
        r = fit_lowrank(labels, targets, rank, init_theta0=init if s < restarts // 2 else None, seed=seed + s, restarts=1, maxiter=3000)
        if best is None or r["mean_fidelity"] > best["mean_fidelity"]:
            best = r
    return best["mean_fidelity"], best["min_fidelity"]


def _job(args):
    name, blocks, targets, regimes = args
    labels = labels_for(blocks)
    row = {"name": name, "blocks": [list(b) for b in blocks], "n_blocks": len(blocks), "K": len(labels)}
    for reg in regimes:
        try:
            mF, minF = evaluate(labels, blocks, targets, reg)
        except ArithmeticError as e:   # MPS conversion failure -> random init only
            mF, minF = float("nan"), float("nan")
        row[f"{reg}_mean"] = mF; row[f"{reg}_min"] = minF
    return row


def candidates():
    fam = []
    for k in range(7, 2, -1):
        fam.append(("A", f"chain{k}", CHAIN[7 - k:]))
    for i in range(7):
        fam.append(("B", f"chain7-minus{CHAIN[i]}", CHAIN[:i] + CHAIN[i + 1:]))
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    for p in pairs:
        for pos in range(8):
            fam.append(("C", f"chain7+{p}@{pos}", CHAIN[:pos] + [p] + CHAIN[pos:]))
    for k in range(1, 8):
        fam.append(("D", f"chain7+sweep{k}", CHAIN + CHAIN[:k]))
        fam.append(("D", f"chain7+sweep{k}msb", CHAIN + CHAIN[7 - k:]))
    fam.append(("D", "chain6+chain6", CHAIN[1:] + CHAIN[1:]))
    fam.append(("D", "chain5+chain5", CHAIN[2:] + CHAIN[2:]))
    for i in range(7):
        for p in pairs:
            if p in CHAIN: continue
            fam.append(("E", f"chain7[{CHAIN[i]}]->{p}", CHAIN[:i] + [p] + CHAIN[i + 1:]))
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--families", default="ABCDE")
    ap.add_argument("--regimes", nargs="+", default=["allvar", "rank1", "fixed"])
    ap.add_argument("--out", default="results/stage1_topology.json")
    args = ap.parse_args()
    dates, S, _ = load_states(2013)
    idx = json.load(open("results/subsets_2013.json"))[str(args.subset)]["indices"]
    targets = S[idx]
    logf = open("logs/stage1_topology.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    fam = [(f, n, b) for f, n, b in candidates() if f in args.families]
    log(f"=== Stage 1 v2: {len(fam)} candidate topologies, regimes {args.regimes}, subset {args.subset}")
    jobs = [(n, b, targets, args.regimes) for f, n, b in fam]
    rows = []
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for (f, n, b), row in zip(fam, ex.map(_job, jobs, chunksize=1)):
            row["family"] = f
            row.update(cx_counts(labels_for(b)))
            rows.append(row)
            log(f"[{f}] {n:30s} blocks={len(b)} cx={row['cx_opt3']:2d} " +
                " ".join(f"{r}:{row[r+'_mean']:.5f}/{row[r+'_min']:.4f}" for r in args.regimes) + f"  ({time.time()-t:.0f}s)")
            json.dump({"subset_indices": idx, "dates": [str(dates[i]) for i in idx], "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
