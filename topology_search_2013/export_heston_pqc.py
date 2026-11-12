#!/usr/bin/env python3
"""Export the validated Heston 2013 U(4) blocks as ONE fixed, explicit PQC.

Dependencies (in your project's .venv):
    python -m pip install 'qiskit>=2,<3'

Run from the heston-nn repository root:
    python -m topology_search_2013.export_heston_pqc

Outputs: a 24-CX / 192-rotation fixed RZ-RY-RZ template, a table of
250 different parameter vectors, OpenQASM 3, and fidelity-verification data.

Crucial endianness: saved blocks use matrix index 2*bit(a)+bit(b), for
pair (a,b). Qiskit's 4x4 Operator on local qubits (0,1) uses index
2*bit(local1)+bit(local0). Thus local qubit 0 maps to GLOBAL b,
local qubit 1 to GLOBAL a. We intentionally use qargs [b,a].

No input data or original optimised unitaries are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from qiskit import QuantumCircuit, qasm3
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import CXGate
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitBasisDecomposer
except ImportError as exc:
    raise SystemExit(
        "Qiskit is required. Activate .venv and run: python -m pip install 'qiskit>=2,<3'"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "topology_search_2013/results/final/M08_7cce5211/unitaries.npz"
DEFAULT_TARGETS = ROOT / "aqce_results_2013/target_states.npz"
DEFAULT_OUTPUT = ROOT / "topology_search_2013/results/pqc_M08_7cce5211"
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
# CNOT(control=local0, target=local1), Qiskit little-endian 2q matrix.
CX01 = np.eye(4, dtype=complex)[:, [0, 3, 2, 1]]


def _slots_from_qiskit(block_circuit: QuantumCircuit):
    """Extract four 1q-gate layers separated by exactly three CXs.

    Each slot is [unitary(local0), unitary(local1)]. All reverse-direction
    CXs are converted to forward-direction CXs by absorbing H into adjacent
    single-qubit layers, without changing the represented unitary.
    """
    slots = [[np.eye(2, dtype=complex) for _ in range(2)] for _ in range(4)]
    directions = []
    layer = 0
    for item in block_circuit.data:
        op = item.operation
        qs = tuple(block_circuit.find_bit(q).index for q in item.qubits)
        if op.name == "cx":
            if layer >= 3 or qs not in ((0, 1), (1, 0)):
                raise ValueError(f"Unexpected two-qubit gate {op.name} on {qs}")
            directions.append(qs)
            layer += 1
        elif len(qs) == 1:
            q = qs[0]
            # Sequential gates act left-multiplied on the state.
            slots[layer][q] = Operator(op).data @ slots[layer][q]
        elif op.name == "barrier":
            continue
        else:
            raise ValueError(f"Unexpected instruction in 2q synthesis: {op.name}")
    if layer != 3:
        raise ValueError(f"Expected exactly 3 CX gates; got {layer}")

    # CX(1 -> 0) == (H x H) CX(0 -> 1) (H x H).
    for j, direction in enumerate(directions):
        if direction == (1, 0):
            for w in range(2):
                slots[j][w] = H @ slots[j][w]
                slots[j + 1][w] = slots[j + 1][w] @ H
    return slots


def _rz(x):
    return np.diag([np.exp(-0.5j * x), np.exp(0.5j * x)])


def _ry(x):
    c, s = np.cos(x / 2), np.sin(x / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _matrix_from_angles(theta_phi_lam):
    theta, phi, lam = theta_phi_lam
    return _rz(phi) @ _ry(theta) @ _rz(lam)


def _block_matrix(theta_phi_lam):
    """Rebuild a block using exactly the PQC's common 3-CX gate template.

    theta_phi_lam shape (4 slots, 2 local wires, 3 Euler angles).
    """
    u = np.eye(4, dtype=complex)
    for slot in range(4):
        m0 = _matrix_from_angles(theta_phi_lam[slot, 0])
        m1 = _matrix_from_angles(theta_phi_lam[slot, 1])
        u = np.kron(m1, m0) @ u
        if slot < 3:
            u = CX01 @ u
    return u


def decompose_block(matrix, twoq, oneq):
    """Return Euler angles (4,2,3), checking U(4) up to global phase."""
    local = twoq(matrix, approximate=False, _num_basis_uses=3)
    slots = _slots_from_qiskit(local)
    angles = np.array(
        [[oneq.angles(m) for m in layer] for layer in slots], dtype=np.float64
    )
    reconstructed = _block_matrix(angles)
    phase_free_overlap = abs(np.trace(matrix.conj().T @ reconstructed)) / 4.0
    if not np.isfinite(phase_free_overlap) or phase_free_overlap < 1 - 5e-9:
        raise ArithmeticError(
            f"Block decomposition changed the unitary: |Tr(U^dag V)|/4 = {phase_free_overlap:.12f}"
        )
    return angles, phase_free_overlap


def make_template(pairs):
    """Return a single QuantumCircuit and its explicit, stable parameter order.

    Each block: 4 local rotation slots of (RZ RY RZ) on both wires,
    interleaved with 3 CXs, always b -> a for saved pair (a,b).
    Thus 8 blocks = 24 CXs + 192 adjustable single-qubit rotations.
    """
    circ = QuantumCircuit(8, name="Heston2013_shared_PQC")
    parameters = []
    for m, (a, b) in enumerate(pairs):
        for slot in range(4):
            for wire, q in enumerate((b, a)):  # IMPORTANT saved matrix ordering
                tag = f"g{m + 1:02d}_s{slot}_w{wire}"
                theta = Parameter(f"{tag}_theta")
                phi = Parameter(f"{tag}_phi")
                lam = Parameter(f"{tag}_lambda")
                parameters.extend((theta, phi, lam))
                # RZ(phi) RY(theta) RZ(lambda) in matrix order.
                circ.rz(lam, int(q))
                circ.ry(theta, int(q))
                circ.rz(phi, int(q))
            if slot < 3:
                circ.cx(int(b), int(a))
        if m < len(pairs) - 1:
            circ.barrier()
    return circ, parameters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-states", type=int, default=None,
                    help="Pilot mode: use a prefix of the dates; default: all")
    ap.add_argument("--no-statevector-check", action="store_true",
                    help="Skip end-to-end 8q fidelity verification (not recommended)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.input) as saved:
        pairs = np.asarray(saved["pairs"], dtype=int)
        matrices = np.asarray(saved["unitaries"], dtype=complex)
        reference_f = np.asarray(saved["fidelities"], dtype=float)
    with np.load(args.targets) as target_data:
        dates = np.asarray(target_data["dates"]).astype(str)
        target_states = np.asarray(target_data["amplitudes"], dtype=complex)

    if matrices.ndim != 4 or matrices.shape[1:] != (len(pairs), 4, 4):
        raise ValueError(f"Unexpected U(4) tensor shape: {matrices.shape}")
    if len(dates) != len(matrices) or len(reference_f) != len(matrices):
        raise ValueError("Input archive and target states disagree about dates")
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("Expected canonical (a<b) topology from the fixed-topology optimiser")
    if args.max_states is not None:
        matrices = matrices[: args.max_states]
        reference_f = reference_f[: args.max_states]
        dates = dates[: args.max_states]
        target_states = target_states[: args.max_states]
    n_states, n_blocks = matrices.shape[:2]
    print(f"Loaded {n_states} dates, {n_blocks} blocks each", flush=True)

    twoq = TwoQubitBasisDecomposer(CXGate(), basis_fidelity=1.0,
                                  euler_basis="ZYZ", pulse_optimize=False)
    oneq = OneQubitEulerDecomposer("ZYZ")
    angles = np.empty((n_states, n_blocks, 4, 2, 3), dtype=np.float64)
    worst_local_overlap = 1.0
    for i in range(n_states):
        for j in range(n_blocks):
            angles[i, j], match = decompose_block(matrices[i, j], twoq, oneq)
            worst_local_overlap = min(worst_local_overlap, match)
        if (i + 1) % 25 == 0 or i == n_states - 1:
            print(f"  Synthesised {i + 1}/{n_states} dates", flush=True)

    circuit, parameters = make_template(pairs)
    flat = angles.reshape(n_states, -1)
    assert flat.shape[1] == len(parameters) == 24 * n_blocks
    assert circuit.count_ops().get("cx", 0) == 3 * n_blocks
    assert circuit.count_ops().get("rz", 0) == 16 * n_blocks
    assert circuit.count_ops().get("ry", 0) == 8 * n_blocks

    actual_f = np.full(n_states, np.nan)
    if not args.no_statevector_check:
        print("Checking all bound PQCs against the saved Heston targets...", flush=True)
        for i in range(n_states):
            bound = circuit.assign_parameters(dict(zip(parameters, flat[i])))
            state = Statevector.from_instruction(bound).data
            actual_f[i] = float(abs(np.vdot(target_states[i], state)) ** 2)
            if abs(actual_f[i] - reference_f[i]) > 2e-7:
                raise AssertionError(
                    f"Date {dates[i]}: saved fidelity {reference_f[i]:.12f}, "
                    f"compiled PQC fidelity {actual_f[i]:.12f}. Export rejected."
                )
        print(f"Verified {n_states}/{n_states} states; min F={actual_f.min():.10f}")

    # Export parameter vectors and their exact ordering. No smoothness objective.
    np.savez_compressed(args.output / "pqc_angles.npz", dates=dates, pairs=pairs,
                        angles=angles, parameter_names=np.array([p.name for p in parameters]),
                        reference_fidelities=reference_f, pqc_fidelities=actual_f)
    with (args.output / "pqc_angles.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "date", "saved_fidelity", "pqc_fidelity"] +
                        [p.name for p in parameters])
        for i in range(n_states):
            writer.writerow([i, dates[i], reference_f[i], actual_f[i]] + flat[i].tolist())

    # This QASM 3 is one parameterised circuit with 192 INPUT ANGLES, not 250
    # independently varying gate arrangements.
    (args.output / "pqc_template.qasm").write_text(qasm3.dumps(circuit))
    (args.output / "pqc_template.txt").write_text(str(circuit.draw(output="text", fold=120)))
    for i in [0, n_states - 1] if n_states > 1 else [0]:
        bound = circuit.assign_parameters(dict(zip(parameters, flat[i])))
        (args.output / f"bound_date_{i:03d}.qasm").write_text(qasm3.dumps(bound))

    report = {
        "description": "ONE fixed 8-qubit PQC; independent angles for each 2013 calibration",
        "saved_U4_matrix_index": "2*bit(a) + bit(b)",
        "qiskit_local_wire_mapping": "local q0 -> global b, local q1 -> global a",
        "pair_order": pairs.tolist(),
        "n_dates": int(n_states),
        "n_blocks": int(n_blocks),
        "n_cx": int(circuit.count_ops().get("cx", 0)),
        "n_rz": int(circuit.count_ops().get("rz", 0)),
        "n_ry": int(circuit.count_ops().get("ry", 0)),
        "n_parameter_angles": len(parameters),
        "min_block_overlap_up_to_global_phase": float(worst_local_overlap),
        "min_reference_fidelity": float(np.min(reference_f)),
        "min_pqc_fidelity": float(np.nanmin(actual_f)) if np.isfinite(actual_f).all() else None,
        "max_abs_fidelity_error": float(np.max(abs(actual_f - reference_f)))
                                   if np.isfinite(actual_f).all() else None,
        "verified_end_to_end": not args.no_statevector_check,
        "note": "24 CX assumes all-to-all connectivity. Hardware routing may add SWAPs.",
    }
    (args.output / "pqc_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved PQC and angles to {args.output}")


if __name__ == "__main__":
    main()
