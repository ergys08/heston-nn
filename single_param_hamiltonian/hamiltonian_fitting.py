#!/usr/bin/env python3
"""Reproduce the explicit 12-term 2013 Heston transport Hamiltonian.

Run from the repository root or any directory:
    python single_param_hamiltonian/hamiltonian_fitting.py --verify

1. Reconstruct Jan 2, 3, 4, 7, 8 state vectors and four FS geodesic tangents.
2. Build the 36 one-Y nearest-neighbour Pauli operators (8 single-qubit
   and 28 two-qubit candidates). All 120 one-Y one-/two-local operators
   are defined, but only chain terms participate in the selection.
3. Joint greedy matching pursuit selects a COMMON set of 12 operators,
   scoring each against all four tangent residuals. Each transition
   gets independent least-squares coefficients during support selection.
4. Freeze the support; fit coefficients using ONLY the Jan 7 -> Jan 8
   tangent, producing H0=sum_k c_k P_k, U(tau)=exp(-i*tau*H0).
5. Test exp(-i*H0)|psi_Jan7> against |psi_Jan8>, and optionally verify
   that the original coefficient values and fidelities are reproduced.

No repository source files are modified by running this code; results are
written to single_param_hamiltonian/results/ (or --out). The tangent
represents the initial direction of a unit-parameter geodesic, not a
physical elapsed-day derivative. The training metrics are in-sample.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from five_initial_state_tangents import DEFAULT_INPUT, DEFAULT_OUTPUT, generate

N_QUBITS = 8
N_BASIS = 256

# This is a validation reference, NEVER used to compute the fitted values.
EXPECTED = {
    "Y4": 0.23147072838961746,
    "Y5Z4": -0.11208196347195332,
    "Y5X6": 0.10610769457998079,
    "Y4Z3": -0.05281443243764007,
    "Y6X5": 0.16247325400214097,
    "Y6Z5": -0.0415468625727334,
    "Y5X4": -0.04018503806278248,
    "Y3": 0.034346961332827955,
    "Y3Z2": -0.019503209226297374,
    "Y4Z5": 0.026358111254297004,
    "Y6": -0.11379959600767128,
    "Y6Z7": -0.020593141006630386,
}


def build_pauli_dictionary():
    """Labels and matrices of -iP, using q1 as the most significant bit.

    All strings have exactly one Y. Hence -iP is a real skew-symmetric
    matrix, mapping the real starting state to a real tangent vector.
    """
    descriptors = [(i, None, None) for i in range(N_QUBITS)]
    for i, j in itertools.combinations(range(N_QUBITS), 2):
        for y, other in ((i, j), (j, i)):
            for axis in ("X", "Z"):
                descriptors.append((y, other, axis))
    source = np.arange(N_BASIS)
    names, skew_matrices, chain_indices = [], [], []
    for y, other, axis in descriptors:
        label = f"Y{y+1}" if other is None else f"Y{y+1}{axis}{other+1}"
        y_mask = 1 << (7-y)
        other_mask = 0 if other is None else 1 << (7-other)
        destination = source ^ y_mask ^ (
            other_mask if axis == "X" else 0
        )
        signs = np.where((source & y_mask) == 0, 1.0, -1.0)
        if axis == "Z":
            signs *= np.where((source & other_mask) == 0, 1.0, -1.0)
        M = csr_matrix(
            (signs, (destination, source)), shape=(N_BASIS, N_BASIS),
            dtype=float,
        )
        if (M+M.T).nnz:
            raise ArithmeticError(f"{label} does not generate a skew matrix")
        names.append(label)
        skew_matrices.append(M)
        if other is None or abs(y-other) == 1:
            chain_indices.append(len(names)-1)
    assert len(names) == 120 and len(chain_indices) == 36
    return names, skew_matrices, chain_indices


def least_squares(matrix, tangent):
    # Match original exploratory calculation, including rank tolerance.
    values, *_ = lstsq(matrix, tangent, cond=1e-12, lapack_driver="gelsy")
    return values


def greedy_joint_support(B, tangents, candidates, n_terms=12):
    """Select a shared support, but refit coefficients separately each day.

    The squared, norm-normalised correlation is summed over the four
    transitions. Greedy selection is heuristic, not a proof of minimality.
    """
    selected = []
    remaining = set(candidates)
    residuals = [v.copy() for v in tangents]
    history = []
    for _ in range(n_terms):
        def score(k):
            return sum(
                float(B[t][:, k] @ residuals[t])**2 /
                max(1e-15, float(B[t][:, k] @ B[t][:, k]))
                for t in range(len(tangents))
            )
        best = max(sorted(remaining), key=score)
        selected.append(best)
        remaining.remove(best)
        coefficients = []
        for t, target in enumerate(tangents):
            c = least_squares(B[t][:, selected], target)
            coefficients.append(c)
            residuals[t] = target - B[t][:, selected] @ c
        mean_residual = np.mean([
            np.linalg.norm(r)/np.linalg.norm(v)
            for r, v in zip(residuals, tangents)
        ])
        history.append({
            "n_terms": len(selected),
            "selected_index": int(best),
            "mean_relative_tangent_error": float(mean_residual),
        })
    return selected, coefficients, history


def exact_transport_fidelity(terms, selected, coefficients, initial, target):
    """U=exp(-iH0), so exp(-iH0)=exp(sum c_k * (-iP_k))."""
    generator = sum(
        (float(c) * terms[k] for k, c in zip(selected, coefficients)),
        csr_matrix((N_BASIS, N_BASIS), dtype=float),
    )
    evolved = expm_multiply(generator, initial)
    if abs(np.linalg.norm(evolved)-1.0) > 1e-9:
        raise ArithmeticError("State norm drift during Hamiltonian evolution")
    return float(np.dot(evolved, target)**2)


def fit_hamiltonian(csv_path=DEFAULT_INPUT):
    dates, states, retained, distances, identities, tangents = generate(csv_path)
    names, matrices, allowed = build_pauli_dictionary()
    B = [
        np.column_stack([M @ states[t] for M in matrices])
        for t in range(len(tangents))
    ]
    chosen, per_date_coefficients, history = greedy_joint_support(
        B, tangents, allowed, n_terms=12
    )

    support = [names[k] for k in chosen]
    training_results = []
    for t, c in enumerate(per_date_coefficients):
        residual = (
            np.linalg.norm(B[t][:, chosen] @ c-tangents[t]) /
            np.linalg.norm(tangents[t])
        )
        fidelity = exact_transport_fidelity(
            matrices, chosen, c, states[t], states[t+1]
        )
        training_results.append({
            "from_date": dates[t],
            "to_date": dates[t+1],
            "identity_fidelity": float(identities[t]),
            "relative_tangent_error": float(residual),
            "transport_fidelity": fidelity,
            **{label: float(value) for label, value in zip(support, c)},
        })

    # The support is chosen jointly on all FOUR transitions; the one
    # Hamiltonian's coefficients are fitted on the LAST transition only.
    frozen = per_date_coefficients[-1]
    result = {
        "dates": dates,
        "n_states": 5,
        "n_transitions": 4,
        "n_all_pauli_candidates": 120,
        "n_chain_candidates": 36,
        "n_selected_operators": 12,
        "selection_method": "greedy joint orthogonal matching pursuit",
        "frozen_coefficient_fit": "2013-01-07 -> 2013-01-08",
        "support": support,
        "coefficients": {k: float(v) for k, v in zip(support, frozen)},
        "frozen_training_relative_tangent_error": float(
            np.linalg.norm(B[-1][:,chosen]@frozen-tangents[-1]) /
            np.linalg.norm(tangents[-1])
        ),
        "frozen_training_transport_fidelity": exact_transport_fidelity(
            matrices, chosen, frozen, states[-2], states[-1]
        ),
        "joint_support_mean_transport_fidelity": float(np.mean(
            [r["transport_fidelity"] for r in training_results]
        )),
        "training_results": training_results,
        "selection_history": history,
        "convention": "H0=sum c_k P_k; U(tau)=exp(-i tau H0); q1 is MSB",
        "warning": "In-sample exploratory model; no future dates used.",
    }
    return result


def verify(result):
    if result["support"] != list(EXPECTED):
        raise AssertionError("Selected operator ordering differs from original")
    error = max(
        abs(result["coefficients"][term]-coeff)
        for term, coeff in EXPECTED.items()
    )
    if error > 1e-7:
        raise AssertionError(
            f"Coefficient mismatch versus original: max difference {error}"
        )
    if abs(
        result["frozen_training_transport_fidelity"]-0.9880941071699827
    ) > 1e-7:
        raise AssertionError("Frozen one-step fidelity mismatch")
    if abs(
        result["joint_support_mean_transport_fidelity"]-0.9958276836957722
    ) > 1e-7:
        raise AssertionError("Mean fitted training fidelity mismatch")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = fit_hamiltonian(args.csv)
    if args.verify:
        diff = verify(result)
        print(f"PASS: reference coefficient max difference = {diff:.3e}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"fitted_hamiltonian.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8",
    )
    with (args.out/"shared_support_training_fits.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "from_date", "to_date", "identity_fidelity",
            "relative_tangent_error", "transport_fidelity",
        ] + result["support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["training_results"])

    print("H0 = sum_k coefficient[k] * Pauli[k]")
    print("U(tau) = exp(-i*tau*H0)")
    for term, coeff in result["coefficients"].items():
        print(f"  {coeff:+.12f}  {term}")
    print(
        "Frozen Jan 7 -> 8 tangent relative error:",
        f"{result['frozen_training_relative_tangent_error']:.9f}",
    )
    print(
        "Frozen Jan 7 -> 8 transport fidelity:",
        f"{result['frozen_training_transport_fidelity']:.9f}",
    )
    print(
        "Mean shared-support training fidelity:",
        f"{result['joint_support_mean_transport_fidelity']:.9f}",
    )
    print("Saved:", args.out/"fitted_hamiltonian.json")
    print("Saved:", args.out/"shared_support_training_fits.csv")


if __name__ == "__main__":
    main()
