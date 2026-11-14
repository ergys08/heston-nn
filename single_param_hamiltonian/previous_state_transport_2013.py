#!/usr/bin/env python3
"""Transport every 2013 Heston state from the preceding historical state.

For each pair (psi_t, psi_(t+1)), fit one scalar:
  tau_t = argmax |<psi_(t+1)|U_Trotter(tau_t)|psi_t>|^2.
The Hamiltonian is the FIXED 12-term fit saved by hamiltonian_fitting.py;
the default circuit is the 18-CNOT, one-step, first-order Trotter product.

The default one-step experiment resets its input to the EXACT previous
historical state. Pass --sequential to additionally test carrying forward
the previous APPROXIMATELY transported state instead. In either case, tau
is optimised against the known next target, so this is not a forecast.

Usage:
  python single_param_hamiltonian/previous_state_transport_2013.py
  python single_param_hamiltonian/previous_state_transport_2013.py --sequential
  python single_param_hamiltonian/previous_state_transport_2013.py --limit 5
  python single_param_hamiltonian/previous_state_transport_2013.py --order 2

Saves a per-transition CSV, summary JSON, and PNG fidelity plot under
single_param_hamiltonian/results. Dependencies: numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from heston.distributions import HestonParameters, discretize_pdf, heston_pdf
from hamiltonian_fitting import build_pauli_dictionary
from trotter_compilation import (
    apply_gates, compile_trotter, gate_counts, load_hamiltonian,
    trotter_schedule,
)

DEFAULT_CSV = ROOT / "heston" / "heston_params_2013.csv"
DEFAULT_HAMILTONIAN = HERE / "results" / "fitted_hamiltonian.json"
DEFAULT_OUT = HERE / "results"


def load_2013_states(path: Path, limit: int = 0):
    """Use precisely the same Heston amplitude encoding as in the fit."""
    dates, states = [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            p = HestonParameters(
                v0=float(row["v0"]), kappa=float(row["kappa"]),
                theta=float(row["theta"]), xi=float(row["sigma"]),
                rho=float(row["rho"]),
            )
            x, pdf, _ = heston_pdf(
                p, n_fft=16384, halfwidth=3.5, maturity=1.0, drift=0.05
            )
            _, _, probabilities, _ = discretize_pdf(
                x, pdf, n_qubits=8, cutoff=2.5
            )
            dates.append(row["date"])
            states.append(np.sqrt(probabilities))
            if limit and len(states) >= limit:
                break
    if len(states) < 2:
        raise ValueError("Expected at least two historical Heston states")
    return dates, np.asarray(states, dtype=float)


def real_generators(terms):
    """Sparse real -i*Pauli matrices with qubit 1 as most significant."""
    names, matrices, _ = build_pauli_dictionary()
    matrix_by_name = dict(zip(names, matrices))
    return [matrix_by_name[name] for name, _ in terms]


def evolve_grid(initial, taus, terms, matrices, schedule):
    """Apply actual Trotter Pauli rotations for many values of tau.

    For a Pauli string with one Y:
      exp(-i a P)|psi> = cos(a)|psi> + sin(a)(-iP)|psi>.
    This equals the compiled Rx/Ry/Rz/CNOT circuit but is faster for
    scalar scans than repeatedly constructing full unitary matrices.
    """
    taus = np.asarray(taus, dtype=float).reshape(-1)
    states = np.broadcast_to(
        np.asarray(initial, dtype=float)[:, None], (len(initial), len(taus))
    ).copy()
    for index, fraction in schedule:
        angle = taus * float(terms[index][1]) * fraction
        states = (
            np.cos(angle)[None, :] * states
            + np.sin(angle)[None, :] * (matrices[index] @ states)
        )
    return states


def fit_scalar(initial, target, terms, matrices, schedule, bound, grid_size):
    """Dense scalar grid followed by local bounded optimisation of top peaks.

    This is numerical one-dimensional maximisation, not a certificate
    of the global optimum for an arbitrary scalar search interval.
    """
    grid = np.linspace(-bound, bound, grid_size)
    overlaps = target @ evolve_grid(
        initial, grid, terms, matrices, schedule
    )
    scores = np.abs(overlaps) ** 2
    index = int(np.argmax(scores))
    best_tau, best_f = float(grid[index]), float(scores[index])
    identity_f = float(abs(np.vdot(target, initial)) ** 2)
    if identity_f > best_f:
        best_tau, best_f = 0.0, identity_f

    peaks = np.flatnonzero(
        (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    ) + 1
    candidates = np.unique(np.r_[0, grid_size - 1, peaks])
    best_indices = candidates[np.argsort(scores[candidates])[-4:]]
    for index in best_indices:
        left = float(grid[max(0, int(index) - 1)])
        right = float(grid[min(grid_size - 1, int(index) + 1)])
        if left >= right:
            continue

        def negative_fidelity(tau):
            evolved = evolve_grid(
                initial, [tau], terms, matrices, schedule
            )[:, 0]
            return -float(abs(np.vdot(target, evolved)) ** 2)

        solution = minimize_scalar(
            negative_fidelity, method="bounded",
            bounds=(left, right), options={"xatol": 1e-10}
        )
        if -float(solution.fun) > best_f:
            best_tau, best_f = float(solution.x), -float(solution.fun)
    return best_tau, min(1.0, best_f), identity_f


def make_plot(path, rows, include_sequential=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = [date.fromisoformat(row["to_date"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.plot(dates, [row["identity_fidelity"] for row in rows],
            label="Identity: exact previous state", alpha=0.85)
    ax.plot(dates, [row["trotter_fidelity"] for row in rows],
            label="Trotter: exact previous state", linewidth=1.7)
    if include_sequential:
        ax.plot(dates, [row["sequential_fidelity"] for row in rows],
                label="Trotter: propagated previous output", linestyle="--")
    ax.set(
        title="2013 historical Heston: previous-state Hamiltonian transport",
        xlabel="Target trading date",
        ylabel="Target-state fidelity",
    )
    ax.set_ylim(0, 1.005)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    if (args.tau_max <= 0 or args.steps < 1 or args.limit < 0
            or args.grid_size < 11 or args.grid_size % 2 != 1):
        raise ValueError(
            "Require tau-max>0, steps>=1, limit>=0 and odd grid-size>=11"
        )
    dates, targets = load_2013_states(args.csv, args.limit)
    terms = load_hamiltonian(args.hamiltonian)
    gates = compile_trotter(terms, order=args.order, steps=args.steps)
    schedule = trotter_schedule(len(terms), args.steps, args.order)
    matrices = real_generators(terms)

    # Make sure fast optimisation and actual compiled gate sequence agree.
    for i in (0, min(3, len(targets)-1)):
        for tau in (-0.7, 0.0, 0.8):
            fast = evolve_grid(
                targets[i], [tau], terms, matrices, schedule
            )[:, 0]
            compiled = apply_gates(gates, targets[i], tau)
            if np.linalg.norm(fast - compiled) > 1e-9:
                raise AssertionError(
                    "Pauli Trotter evolution disagrees with compiled gates"
                )

    rows = []
    propagated_state = targets[0].copy()
    for t in range(1, len(dates)):
        tau, fidelity, identity = fit_scalar(
            targets[t-1], targets[t], terms, matrices, schedule,
            args.tau_max, args.grid_size,
        )
        row = {
            "from_date": dates[t-1],
            "to_date": dates[t],
            "best_tau": tau,
            "trotter_fidelity": fidelity,
            "identity_fidelity": identity,
            "improvement_over_identity": fidelity - identity,
        }
        if args.sequential:
            seq_tau, seq_f, seq_identity = fit_scalar(
                propagated_state, targets[t], terms, matrices, schedule,
                args.tau_max, args.grid_size,
            )
            propagated_state = evolve_grid(
                propagated_state, [seq_tau], terms, matrices, schedule
            )[:, 0]
            propagated_state /= np.linalg.norm(propagated_state)
            row.update({
                "sequential_tau": seq_tau,
                "sequential_fidelity": seq_f,
                "sequential_identity_fidelity": seq_identity,
            })
        rows.append(row)
        if t % 25 == 0 or t == len(dates) - 1:
            print(f"Completed {t}/{len(dates)-1} transitions", flush=True)

    values = np.array([r["trotter_fidelity"] for r in rows])
    identities = np.array([r["identity_fidelity"] for r in rows])
    summary = {
        "n_states": len(dates),
        "n_transitions": len(rows),
        "from": dates[0],
        "through": dates[-1],
        "method": "fixed Hamiltonian, one scalar fitted per transition",
        "initial_state_mode": "exact previous historical Heston state",
        "order": args.order,
        "steps": args.steps,
        "gate_counts": gate_counts(gates),
        "tau_range": [-args.tau_max, args.tau_max],
        "mean_transport_fidelity": float(values.mean()),
        "median_transport_fidelity": float(np.median(values)),
        "minimum_transport_fidelity": float(values.min()),
        "minimum_fidelity_date": rows[int(np.argmin(values))]["to_date"],
        "mean_identity_fidelity": float(identities.mean()),
        "mean_gain_over_identity": float((values - identities).mean()),
        "n_fidelity_ge_0_99": int(np.sum(values >= 0.99)),
        "n_fidelity_ge_0_999": int(np.sum(values >= 0.999)),
        "warning": "Scalar fits use next exact target; Hamiltonian and gate topology are fixed.",
    }
    if args.sequential:
        sequential = np.array([r["sequential_fidelity"] for r in rows])
        summary.update({
            "sequential_mean_fidelity": float(sequential.mean()),
            "sequential_minimum_fidelity": float(sequential.min()),
            "sequential_final_fidelity": float(sequential[-1]),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"previous_state_transport_2013_order{args.order}_steps{args.steps}"
    if args.limit:
        stem += f"_first{len(dates)}"
    if args.sequential:
        stem += "_with_sequential"
    csv_file = args.out / f"{stem}.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_file = args.out / f"{stem}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    figure_file = args.out / f"{stem}.png"
    make_plot(figure_file, rows, args.sequential)
    print(json.dumps(summary, indent=2))
    print("CSV:", csv_file)
    print("Summary:", summary_file)
    print("Figure:", figure_file)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--hamiltonian", type=Path, default=DEFAULT_HAMILTONIAN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--grid-size", type=int, default=801)
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Quick test: use first N dates; 0 means all dates")
    p.add_argument("--sequential", action="store_true",
                   help="Additionally propagate previous approximate state")
    run(p.parse_args())


if __name__ == "__main__":
    main()
