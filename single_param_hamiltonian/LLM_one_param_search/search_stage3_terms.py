"""Stage 3: term-level refinement of a block architecture.

(a) Pruning: remove fixed generators one at a time (least loss first) while
    the mean fidelity drop stays below a tolerance -> fewer Hamiltonian
    terms; CNOTs only fall if an entire block disappears.
(b) Higher-order terms: try inserting each 3-local odd-Y string (on triples
    of qubits touched by the architecture) at every block boundary; compare
    the fidelity gain per added CNOT against adding another pair block.

Usage: python search_stage3_terms.py --arch results/stage2_<tag>_subset8.json --n-var 2
"""
from __future__ import annotations
import argparse, itertools, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ansatz import fit
from compile import cx_counts
from paulis import compact_label, odd_y_strings, support
from pool import record, discard
from states import load_states


def _refit(args):
    labels, targets, variable, init_fixed, init_var, seed, restarts = args
    r = fit(labels, targets, variable=variable, init_fixed=init_fixed, init_var=init_var,
            restarts=restarts, seed=seed, maxiter=1500, perturb=0.2)
    return r


def warm_inits(current, new_labels):
    """Map a FitResult onto a new label list (missing labels start at 0)."""
    theta = current.theta_full()
    old = {l: i for i, l in enumerate(current.labels)}
    fixed = [l for l in new_labels if l not in current.variable]
    var = [l for l in new_labels if l in current.variable]
    init_fixed = np.array([theta[0, old[l]] if l in old else 0.0 for l in fixed])
    init_var = np.stack([theta[:, old[l]] if l in old else np.zeros(theta.shape[0]) for l in var], 1) if var else None
    return init_fixed, init_var, var


def prune(labels, variable, targets, current, tol, workers, log):
    """Backward elimination of fixed generators."""
    labels = list(labels)
    history = []
    while True:
        removable = [l for l in labels if l not in variable]
        jobs = []
        for l in removable:
            new = [x for x in labels if x != l]
            init_fixed, init_var, var = warm_inits(current, new)
            jobs.append((new, targets, var, init_fixed, init_var, 0, 2))
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_refit, jobs, chunksize=2))
        best_i = int(np.argmax([o.mean_fidelity for o in out]))
        drop = current.mean_fidelity - out[best_i].mean_fidelity
        if drop > tol:
            log(f"  prune stop: removing {compact_label(removable[best_i])} would cost {drop:.2e} > tol")
            break
        labels = [x for x in labels if x != removable[best_i]]
        current = out[best_i]
        history.append({"removed": removable[best_i], "mean_F": current.mean_fidelity, "min_F": current.min_fidelity, "K": len(labels)})
        log(f"  pruned {compact_label(removable[best_i]):6s} -> K={len(labels)} meanF {current.mean_fidelity:.6f} minF {current.min_fidelity:.5f}")
    return labels, current, history


def three_local_candidates(labels):
    qubits = sorted({q for l in labels for q in support(l)})
    cands = []
    for triple in itertools.combinations(qubits, 3):
        cands += odd_y_strings(triple)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--n-var", type=int, default=2)
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--skip-3local", action="store_true")
    ap.add_argument("--max-span", type=int, default=3, help="max qubit span of 3-local candidates")
    args = ap.parse_args()
    arch = json.load(open(args.arch))
    labels = arch["labels"]
    variable = arch["curve"][args.n_var]["variable"]
    dates, S, _ = load_states(2013)
    idx = json.load(open("results/subsets_2013.json"))[str(args.subset)]["indices"]
    targets = S[idx]
    tag = args.arch.split("stage2_")[1].rsplit("_subset", 1)[0] + f"_v{args.n_var}"
    logf = open("logs/stage3_terms.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    coeffs = arch["fits"][str(args.n_var)]
    init_fixed = np.array([coeffs["fixed"][l] for l in labels if l not in variable])
    init_var = np.array([coeffs["variable"][l] for l in variable]).T if variable else None
    current = fit(labels, targets, variable=variable, init_fixed=init_fixed, init_var=init_var, restarts=3, maxiter=2000)
    cx = cx_counts(labels)
    log(f"=== Stage 3 on {tag}: K={len(labels)} n_var={len(variable)} cx={cx['cx_opt3']} meanF {current.mean_fidelity:.6f} minF {current.min_fidelity:.5f}")

    # (a) prune
    pruned_labels, pruned, hist = prune(labels, variable, targets, current, args.tol, args.workers, log)
    cxp = cx_counts(pruned_labels)
    record("stage3-pruned", pruned_labels, variable, idx, pruned, cxp, notes=f"pruned from {tag} tol={args.tol}")
    log(f"  pruned architecture: K={len(pruned_labels)} cx={cxp['cx_opt3']} meanF {pruned.mean_fidelity:.6f} minF {pruned.min_fidelity:.5f}")
    out = {"source": args.arch, "n_var": args.n_var, "variable": variable, "start": {"labels": labels, "mean_F": current.mean_fidelity, "min_F": current.min_fidelity, **cx},
           "pruned": {"labels": pruned_labels, "mean_F": pruned.mean_fidelity, "min_F": pruned.min_fidelity, **cxp,
                      "coefficients": pruned.coefficients_dict()}, "prune_history": hist}

    # (b) 3-local insertions (fixed coefficient) at each block boundary
    if not args.skip_3local:
        cands = [c for c in three_local_candidates(pruned_labels) if support(c)[-1] - support(c)[0] <= args.max_span]
        positions = sorted({0, len(pruned_labels)} | {i for i in range(1, len(pruned_labels)) if support(pruned_labels[i]) != support(pruned_labels[i-1]) and len(support(pruned_labels[i])) == 2})
        log(f"  3-local search: {len(cands)} strings x {len(positions)} positions")
        jobs, meta = [], []
        for c in cands:
            for p in positions:
                new = pruned_labels[:p] + [c] + pruned_labels[p:]
                init_fixed, init_var, var = warm_inits(pruned, new)
                jobs.append((new, targets, var, init_fixed, init_var, 0, 2)); meta.append((c, p))
        t = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_refit, jobs, chunksize=4))
        order = np.argsort([-r.mean_fidelity for r in res])
        rows = []
        for i in order[:15]:
            c, p = meta[i]; new = pruned_labels[:p] + [c] + pruned_labels[p:]
            cxn = cx_counts(new)
            rows.append({"term": c, "compact": compact_label(c), "position": p, "mean_F": res[i].mean_fidelity, "min_F": res[i].min_fidelity,
                         "gain": res[i].mean_fidelity - pruned.mean_fidelity, **cxn})
            log(f"    +{compact_label(c):8s} @{p:2d}: meanF {res[i].mean_fidelity:.6f} (+{res[i].mean_fidelity - pruned.mean_fidelity:.2e}) cx_opt3 {cxn['cx_opt3']} ({time.time()-t:.0f}s)")
        # also try making the best 3-local term's coefficient variable
        c, p = meta[order[0]]; new = pruned_labels[:p] + [c] + pruned_labels[p:]
        init_fixed, init_var, var = warm_inits(pruned, new)
        r3 = fit(new, targets, variable=var + [c], init_fixed=init_fixed[[i for i, l in enumerate([x for x in new if x not in var]) if l != c]],
                 init_var=np.concatenate([init_var, np.zeros((len(idx), 1))], 1) if init_var is not None else np.zeros((len(idx), 1)), restarts=3, maxiter=1500)
        log(f"    best 3-local term {compact_label(c)} made variable too: meanF {r3.mean_fidelity:.6f} minF {r3.min_fidelity:.5f}")
        out["three_local"] = rows
        out["three_local_variable"] = {"term": c, "mean_F": r3.mean_fidelity, "min_F": r3.min_fidelity}
        record("stage3-3local", new, var, idx, res[order[0]], cx_counts(new), notes=f"best 3-local insertion into pruned {tag}")
    json.dump(out, open(f"results/stage3_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
