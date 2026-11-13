#!/usr/bin/env python3
"""Run the real TTNOpt FactorizeTensor topology optimiser on 2013 Heston amplitudes.

Run from any directory:
  python "tensor network/run_ttnopt_2013.py" --sweeps 4 --bonds 2 4

Unlike an exhaustive Schmidt-spectrum topology proxy, this constructs/optimises
actual TTN tensors using TTNOpt, then computes the represented state's fidelity.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heston.distributions import HestonParameters, heston_pdf, discretize_pdf
from ttnopt.src import FactorizeTensor, TreeTensorNetwork

BASE = ROOT / "tensor network" / "ttnopt_results_2013"
PARAMS = ROOT / "heston" / "heston_params_2013.csv"


def load_states():
    out = []
    with PARAMS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            params = HestonParameters(
                v0=float(row["v0"]), kappa=float(row["kappa"]),
                theta=float(row["theta"]), xi=float(row["sigma"]),
                rho=float(row["rho"]),
            )
            x, pdf, _ = heston_pdf(params)
            _, _, p, _ = discretize_pdf(x, pdf, n_qubits=8, cutoff=2.5)
            v = np.sqrt(p).reshape((2,) * 8)
            assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-12)
            out.append((row["date"], v))
    if len(out) != 250:
        raise ValueError(f"Expected 250 historical states; got {len(out)}")
    return out


def graph_splits(edges, n=8):
    """Canonical internal-edge splits, independent of internal edge numbering."""
    edges = [list(map(int, x)) for x in edges]
    occurrences = {}
    for tid, tri in enumerate(edges):
        for edge in tri:
            occurrences.setdefault(edge, []).append(tid)
    splits = set()
    for bond, owners in occurrences.items():
        if len(owners) != 2:
            continue
        seen, todo, subset = set(), [owners[0]], set()
        while todo:
            tid = todo.pop()
            if tid in seen:
                continue
            seen.add(tid)
            for edge in edges[tid]:
                if edge == bond:
                    continue
                ts = occurrences[edge]
                if len(ts) == 1:
                    if not 0 <= edge < n:
                        raise ValueError(f"Unexpected physical edge {edge}")
                    subset.add(edge)
                elif len(ts) == 2:
                    todo.extend(x for x in ts if x != tid)
                else:
                    raise ValueError(f"Invalid edge multiplicity: {edge}: {ts}")
        if len(subset) < 2 or len(subset) > n - 2:
            continue
        complement = set(range(n)) - subset
        a, b = tuple(sorted(subset)), tuple(sorted(complement))
        canonical = min(a, b)
        splits.add(canonical)
    if len(splits) != n - 3:
        raise ValueError(f"Expected {n-3} internal splits, found {len(splits)}: {splits}")
    return tuple(sorted(splits))


def fit_one(target, chi, sweeps):
    initial = TreeTensorNetwork.mps(
        target.ndim, target=target, max_bond_dimension=chi
    )
    opt = FactorizeTensor(initial, max_bond_dim=chi)
    baseline = float(np.real(opt.get_fidelity(target)))
    # Suppress thousands of fidelity print lines in TTNOpt's sweep implementation.
    with open("/dev/null", "w") as quiet, contextlib.redirect_stdout(quiet):
        opt.run(
            target=target, opt_fidelity=True, opt_structure=1,
            max_num_sweep=sweeps, temperature=0.0, tau=1,
            fidelity_convergence_threshold=1e-10,
            entanglement_convergence_threshold=1e-10,
            max_truncation_error=0.0,
        )
    fidelity = float(np.real(opt.get_fidelity(target)))
    if not all(np.isfinite(z) for z in (baseline, fidelity)):
        raise ArithmeticError("Nonfinite TTNOpt fidelity")
    if baseline < -1e-6 or baseline > 1 + 1e-6 or fidelity < -1e-6 or fidelity > 1 + 1e-6:
        raise ArithmeticError(f"Invalid fidelity: baseline {baseline}, final {fidelity}")
    edges = [list(map(int, e)) for e in opt.psi.edges]
    splits = graph_splits(edges, n=target.ndim)
    return baseline, fidelity, edges, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonds", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--out", type=Path, default=BASE)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=250)
    args = ap.parse_args()
    if any(b < 2 for b in args.bonds) or args.sweeps < 1:
        raise ValueError("Bond dimensions must be >= 2; sweeps must be >= 1")
    out = args.out.resolve()
    graphs = out / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    states = load_states()[args.start:args.stop]
    if not states:
        raise ValueError("Selected date range is empty")
    results, failures = [], []
    for date, state in states:
        for chi in args.bonds:
            try:
                initial, fidelity, edges, splits = fit_one(state, chi, args.sweeps)
                relative_graph = f"graphs/{date}_chi{chi}.json"
                with (out / relative_graph).open("w") as f:
                    json.dump({"date": date, "bond_dimension": chi,
                               "edges": edges,
                               "splits_zero_based": [list(s) for s in splits],
                               "splits_qubits_one_based": [[x+1 for x in s] for s in splits],
                               "initial_fidelity": initial, "fidelity": fidelity},
                              f, indent=2)
                    f.write("\n")
                signature = "|".join(
                    "-".join(str(x + 1) for x in group) for group in splits
                )
                results.append({"date": date, "bond_dimension": chi,
                                "initial_fidelity": initial, "fidelity": fidelity,
                                "infidelity": 1 - fidelity,
                                "topology_signature": signature,
                                "graph_file": relative_graph})
                print(f"{date} chi={chi} F={fidelity:.10g} signature={signature}", flush=True)
            except Exception as exc:
                failures.append({"date": date, "bond_dimension": chi,
                                 "error": repr(exc), "traceback": traceback.format_exc()})
                print(f"FAILED {date} chi={chi}: {exc}", flush=True)
    fields = ["date", "bond_dimension", "initial_fidelity", "fidelity",
              "infidelity", "topology_signature", "graph_file"]
    with (out / "per_date_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "algorithm": "TTNOpt FactorizeTensor.run, opt_fidelity=True, opt_structure=1",
        "meaning": "Locally optimised TTN geometry and TTN tensor amplitudes, not a global minimum or an optimal CNOT circuit",
        "source": str(PARAMS.relative_to(ROOT)),
        "n_target_states": len(states),
        "n_attempted": len(states) * len(args.bonds),
        "n_successes": len(results),
        "bonds": args.bonds, "sweeps": args.sweeps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ttnopt_version": importlib.metadata.version("ttnopt"),
        "qubit_axes": "axis 0 (q1) is most significant bit; axis 7 (q8) is least significant bit",
        "by_bond_dimension": {},
        "failures": failures,
    }
    for chi in args.bonds:
        r = [x for x in results if x["bond_dimension"] == chi]
        if not r:
            continue
        counter = Counter(x["topology_signature"] for x in r)
        summary["by_bond_dimension"][str(chi)] = {
            "n": len(r), "n_unique_topologies": len(counter),
            "top_topologies": [{"signature": sig, "count": ct}
                               for sig, ct in counter.most_common(20)],
            "mean_fidelity": float(np.mean([x["fidelity"] for x in r])),
            "min_fidelity": float(min(x["fidelity"] for x in r)),
            "mean_initial_fidelity": float(np.mean([x["initial_fidelity"] for x in r])),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} TTNOpt optimisations failed; see summary.json")
    print("TTNOpt completed:", len(results), "optimisations", flush=True)


if __name__ == "__main__":
    main()
