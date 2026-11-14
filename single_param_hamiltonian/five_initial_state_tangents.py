#!/usr/bin/env python3
"""Reconstruct the first five 2013 Heston states and four geodesic tangents.

Run from a checkout of heston-nn:
    python single_param_hamiltonian/five_initial_state_tangents.py

The input is heston/heston_params_2013.csv, in its existing row order.
The output directory defaults to single_param_hamiltonian/results/.

The tangent v_t is the initial velocity of the shortest great-circle path
from psi_t to psi_(t+1), for one unit of interpolation parameter. It is NOT
a derivative with respect to real elapsed time or Heston model parameters.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heston.distributions import HestonParameters, discretize_pdf, heston_pdf

DEFAULT_INPUT = REPO_ROOT / "heston" / "heston_params_2013.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def reconstruct_first_five_states(csv_path: Path = DEFAULT_INPUT):
    """Encode five PDFs as 8-qubit sqrt-probability state vectors.

    FFT size 16384, halfwidth 3.5, cutoff 2.5, 256 bins, maturity 1.0,
    drift 0.05. Qubit 1 is the most significant basis-index bit.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:5]
    if len(rows) != 5:
        raise ValueError(f"Expected at least five CSV rows; found {len(rows)}")
    dates, states, retained_masses = [], [], []
    for row in rows:
        params = HestonParameters(
            v0=float(row["v0"]),
            kappa=float(row["kappa"]),
            theta=float(row["theta"]),
            xi=float(row["sigma"]),
            rho=float(row["rho"]),
        )
        x, pdf, _ = heston_pdf(
            params, n_fft=16384, halfwidth=3.5, maturity=1.0, drift=0.05
        )
        _, _, probabilities, retained = discretize_pdf(
            x, pdf, n_qubits=8, cutoff=2.5
        )
        psi = np.sqrt(probabilities).astype(np.float64)
        if psi.shape != (256,) or not np.isclose(
            np.linalg.norm(psi), 1.0, atol=1e-12
        ):
            raise ArithmeticError(f"Invalid state for {row['date']}")
        dates.append(row["date"])
        states.append(psi)
        retained_masses.append(float(retained))
    return dates, np.stack(states), np.asarray(retained_masses)


def geodesic_tangents(states: np.ndarray):
    """Return distances, identity fidelities, and FS geodesic tangent vectors.

    d = arccos(<psi|target>).
    v = d/sin(d) * (target - cos(d)*psi).
    Because states are real nonnegative, their phases are already aligned.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 256:
        raise ValueError("Expected a matrix of 256-amplitude state vectors")
    distances, fidelities, tangents = [], [], []
    for psi, target in zip(states[:-1], states[1:]):
        overlap = float(np.clip(np.dot(psi, target), -1.0, 1.0))
        if overlap < 0:
            raise ValueError("Input state phases need aligning")
        distance = math.acos(overlap)
        factor = distance / math.sin(distance) if distance > 1e-10 else 1.0
        tangent = factor * (target - overlap * psi)
        if abs(np.dot(psi, tangent)) > 1e-9:
            raise ArithmeticError("Tangent should be orthogonal to initial state")
        distances.append(distance)
        fidelities.append(overlap**2)
        tangents.append(tangent)
    return np.asarray(distances), np.asarray(fidelities), np.stack(tangents)


def generate(csv_path: Path = DEFAULT_INPUT):
    dates, states, masses = reconstruct_first_five_states(csv_path)
    distances, fidelities, tangents = geodesic_tangents(states)
    return dates, states, masses, distances, fidelities, tangents


def save_results(output_dir: Path, dates, states, masses, distances,
                 fidelities, tangents):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "five_initial_state_tangents.npz",
        dates=np.asarray(dates),
        states=states,
        tangents=tangents,
        fs_distances=distances,
        identity_fidelities=fidelities,
        retained_pdf_mass=masses,
    )
    with (out / "five_initial_state_tangents.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "from_date", "to_date", "fs_distance_radians",
            "identity_fidelity", "tangent_norm"
        ])
        for i in range(4):
            writer.writerow([
                dates[i], dates[i+1],
                f"{distances[i]:.17g}",
                f"{fidelities[i]:.17g}",
                f"{np.linalg.norm(tangents[i]):.17g}",
            ])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = generate(args.csv)
    output = save_results(args.out, *data)
    dates, _, masses, distances, fidelities, tangents = data
    print("Dates:", ", ".join(dates))
    print("FFT=16384, halfwidth=3.5, cutoff=2.5, n_qubits=8")
    print("Transition                   FS distance     Identity F       ||tangent||")
    for i in range(4):
        print(
            f"{dates[i]} -> {dates[i+1]}    {distances[i]:.9f}"
            f"       {fidelities[i]:.9f}      {np.linalg.norm(tangents[i]):.9f}"
        )
    print(
        f"PDF retained-mass range: [{masses.min():.10f},"
        f" {masses.max():.10f}]"
    )
    print("Saved:", output / "five_initial_state_tangents.npz")
    print("Saved:", output / "five_initial_state_tangents.csv")


if __name__ == "__main__":
    main()
