"""Stage 2: for a given block topology, decide WHICH coefficients must vary.

Forward greedy selection over coefficient POSITIONS: start from the all-fixed
fit (warm-started from the MPS solution of the mean state) and repeatedly
release the single coefficient whose per-state freedom most increases mean
fidelity, refitting everything else.  The low-rank "direction" model
(rank 1..3: a few scalars per state that each move many angles) and the
all-variable bound are fitted for comparison.

Usage: python search_stage2_variable.py --blocks "6,7 5,6 4,5 3,4 2,3 1,2 0,1" --subset 8 --max-var 8
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ansatz import fit, fit_lowrank
from compile import cx_counts
from paulis import compact_label
from pool import record
from search_stage1_topology import labels_for, mps_theta_for, CHAIN
from states import load_states


def _try_var(args):
    labels, targets, variable, init_fixed, init_var, seed = args
    r = fit(labels, targets, variable=variable, init_fixed=init_fixed, init_var=init_var,
            restarts=3, seed=seed, maxiter=1500, perturb=0.2)
    return variable[-1], r


def all_fixed_fit(labels, blocks, targets, restarts=8, seed=3):
    mean = targets.mean(0); mean /= np.linalg.norm(mean)
    init = mps_theta_for(blocks, mean, 2 if len(blocks) > 7 else 1)
    return fit(labels, targets, variable=[], init_fixed=init, restarts=restarts, seed=seed, maxiter=2000, init_scale=0.5, perturb=0.2)


def forward_variable_selection(labels, targets, max_var, workers, log, base, candidates=None):
    curve = [{"n_variable": 0, "variable": [], "mean_F": base.mean_fidelity, "min_F": base.min_fidelity}]
    fits = {0: base}
    current = base
    candidates = list(range(len(labels))) if candidates is None else list(candidates)
    for m in range(1, max_var + 1):
        t = time.time()
        theta = current.theta_full()
        jobs = []
        for cand in candidates:
            if cand in current.variable:
                continue
            variable = sorted(current.variable + [cand])
            fixed_pos = [i for i in range(len(labels)) if i not in variable]
            jobs.append((labels, targets, variable[:-1] + [cand] if False else variable, theta[0, fixed_pos], theta[:, variable], m))
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_try_var, jobs, chunksize=1))
        # identify which candidate each result corresponds to
        scored = []
        for (labels_, _, variable, _, _, _), (_, r) in zip(jobs, out):
            new = [v for v in variable if v not in current.variable][0]
            scored.append((new, r))
        scored.sort(key=lambda o: -o[1].mean_fidelity)
        best_pos, current = scored[0]
        fits[m] = current
        top = [(f"{p}:{compact_label(labels[p])}", round(r.mean_fidelity, 6)) for p, r in scored[:5]]
        curve.append({"n_variable": m, "variable": list(current.variable), "variable_labels": current.variable_labels,
                      "mean_F": current.mean_fidelity, "min_F": current.min_fidelity, "top_candidates": top})
        log(f"  {m} variable: + {best_pos}:{compact_label(labels[best_pos]):6s} meanF {current.mean_fidelity:.6f} minF {current.min_fidelity:.5f}"
            f"  runners-up {top[1:4]} ({time.time()-t:.0f}s)")
    return curve, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="chain7", help='"chain7" / "chain6" ... or e.g. "6,7 5,6 0,1"')
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--max-var", type=int, default=8)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.blocks.startswith("chain"):
        k = int(args.blocks[5:]); blocks = CHAIN[7 - k:]
    else:
        blocks = [tuple(int(x) for x in b.split(",")) for b in args.blocks.split()]
    labels = labels_for(blocks)
    dates, S, _ = load_states(2013)
    idx = json.load(open("results/subsets_2013.json"))[str(args.subset)]["indices"]
    targets = S[idx]
    tag = args.tag or (args.blocks if args.blocks.startswith("chain") else "b" + "_".join(f"{i}{j}" for i, j in blocks))
    logf = open("logs/stage2_variable.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    cx = cx_counts(labels)
    log(f"=== Stage 2 variable selection: {tag} blocks {blocks} K={len(labels)} cx_opt3={cx['cx_opt3']} subset {args.subset}")
    t = time.time()
    base = all_fixed_fit(labels, blocks, targets)
    log(f"  all-fixed: meanF {base.mean_fidelity:.6f} minF {base.min_fidelity:.5f} ({time.time()-t:.0f}s)")
    curve, fits = forward_variable_selection(labels, targets, args.max_var, args.workers, log, base)
    for m, r in fits.items():
        record("stage2", labels, r.variable, idx, r, cx, notes=f"topology {tag}; forward greedy variable selection")
    lowrank = []
    mean = targets.mean(0); mean /= np.linalg.norm(mean)
    for rank in (1, 2, 3):
        best = None
        for s in range(6):
            r = fit_lowrank(labels, targets, rank, init_theta0=fits[0].theta_fixed if s < 3 else None, seed=s, restarts=1, maxiter=3000)
            if best is None or r["mean_fidelity"] > best["mean_fidelity"]:
                best = r
        lowrank.append({"rank": rank, "mean_F": best["mean_fidelity"], "min_F": best["min_fidelity"],
                        "theta0": best["theta0"].tolist(), "dirs": best["dirs"].tolist(), "tau": best["tau"].tolist()})
        log(f"  low-rank {rank}: meanF {best['mean_fidelity']:.6f} minF {best['min_fidelity']:.5f}")
    init = np.stack([mps_theta_for(blocks, tg, 2 if len(blocks) > 7 else 1) for tg in targets])
    allv = fit(labels, targets, variable=list(range(len(labels))), init_var=init, restarts=4, seed=1, maxiter=2000, init_scale=0.5)
    log(f"  all-variable: meanF {allv.mean_fidelity:.6f} minF {allv.min_fidelity:.5f}")
    out = {"blocks": [list(b) for b in blocks], "labels": labels, "cx": cx, "subset": idx, "curve": curve,
           "lowrank": lowrank, "allvar": {"mean_F": allv.mean_fidelity, "min_F": allv.min_fidelity},
           "fits": {str(m): r.coefficients_dict() for m, r in fits.items()}}
    json.dump(out, open(f"results/stage2_{tag}_subset{args.subset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
