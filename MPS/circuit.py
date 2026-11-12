"""Approximate iterative MPS preparation in the style of Algorithm 2.

Reference: Iaconis, Johri & Zhu (2024), npj Quantum Information 10, 15,
https://doi.org/10.1038/s41534-024-00805-0. The iterative χ=2 layer
method is credited there to their reference [26].

For a normalized target |psi_0> and requested layer count D:
    |phi_i> = right-to-left SVD bond-2 approximation of |psi_i>
    construct U_i such that U_i |0...0> = |phi_i>
    |psi_{i+1}> = U_i^dagger |psi_i>
    preparation circuit = U_0 U_1 ... U_(D-1)

A single layer contains n-1 nearest-neighbour 4x4 isometric completions
plus one final one-qubit unitary. The Qiskit circuit executes layers in
reverse extraction order. Matrix gates use Qiskit little-endian ordering.

The *classical* residual is a dense statevector: this is well suited to
8-qubit Heston targets but does not inherit the paper's large-n MPS
contraction scaling. No variational post-optimisation is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv

import numpy as np

from .mps_decomposition import decompose_state, normalize_state


@dataclass(frozen=True)
class LocalGate:
    qubits: tuple[int, ...]  # q0 is least-significant / local q0 for pair
    unitary: np.ndarray


@dataclass(frozen=True)
class MPSLayer:
    """Preparation gates in forward application order."""

    gates: tuple[LocalGate, ...]
    mps_fidelity: float  # |<residual | normalized chi=2 MPS>|**2
    bonds: tuple[int, ...]


@dataclass(frozen=True)
class IterativeMPSResult:
    layers: tuple[MPSLayer, ...]  # extraction order (outermost first)
    fidelities: tuple[float, ...]  # target fidelity after 1,...,D layers
    n_qubits: int

    @property
    def fidelity(self) -> float:
        return self.fidelities[-1]

    @property
    def n_two_qubit_gates(self) -> int:
        return sum(len(l.gates) - 1 for l in self.layers)

    @property
    def nominal_real_cx_count(self) -> int:
        """Paper's 2 CNOT per real SO(4) gate, not a compiler measurement."""
        return 2 * self.n_two_qubit_gates

    def prepare_statevector(self) -> np.ndarray:
        state = np.zeros(1 << self.n_qubits, dtype=complex)
        state[0] = 1
        for layer in reversed(self.layers):
            for gate in layer.gates:
                state = apply_local_unitary(state, gate.unitary, gate.qubits)
        return state

    def to_qiskit(self, *, decompose_to_cx: bool = False):
        """Return an explicit Qiskit circuit, optionally compiling to CX + 1q.

        Qiskit is an optional dependency; internal simulation needs NumPy only.
        CX totals depend on the installed Qiskit synthesis version and should
        be read from circuit.count_ops(), not assumed from the paper.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError("Install qiskit to export a QuantumCircuit") from e
        circuit = QuantumCircuit(self.n_qubits, name="iterative_mps")
        if decompose_to_cx:
            from qiskit.synthesis import two_qubit_cnot_decompose, OneQubitEulerDecomposer
            oneq = OneQubitEulerDecomposer("ZYZ")
        for li, layer in enumerate(reversed(self.layers)):
            for gate in layer.gates:
                if decompose_to_cx:
                    sub = (two_qubit_cnot_decompose(gate.unitary, approximate=False)
                           if len(gate.qubits) == 2 else oneq(gate.unitary))
                    circuit.compose(sub, qubits=list(gate.qubits), inplace=True)
                else:
                    circuit.unitary(gate.unitary, list(gate.qubits),
                                    label="MPS2" if len(gate.qubits) == 2 else "MPS1")
            if li != len(self.layers) - 1:
                circuit.barrier()
        return circuit


def _complete_isometry(first_columns: np.ndarray) -> np.ndarray:
    """Extend supplied orthonormal columns to a square unitary.

    For real columns, choose determinant +1 by flipping the final *free*
    column. This does not change the action on the incoming MPS bond space.
    """
    w = np.asarray(first_columns)
    d, k = w.shape
    if not 0 < k <= d:
        raise ValueError("Isometry must have 1..d columns")
    if not np.allclose(w.conj().T @ w, np.eye(k), atol=1e-9):
        raise ValueError("Supplied MPS core is not right-isometric")
    basis = [w[:, i].copy() for i in range(k)]
    real = not np.iscomplexobj(w)
    for e in np.eye(d, dtype=w.dtype).T:
        if len(basis) == d:
            break
        v = e.copy()
        for _ in range(2):  # modified Gram-Schmidt + re-orthogonalization
            for col in basis:
                v -= col * np.vdot(col, v)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)
    if len(basis) != d:
        raise ArithmeticError("Failed to complete MPS isometry")
    u = np.column_stack(basis)
    if real and k < d and np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if not np.allclose(u.conj().T @ u, np.eye(d), atol=1e-9):
        raise ArithmeticError("Isometry completion was not unitary")
    return u


def layer_from_mps(cores: tuple[np.ndarray, ...], *,
                   approximation_fidelity: float = 1.0) -> MPSLayer:
    """Construct n-1 nearest-neighbour 2q gates + one 1q gate.

    Right-canonical site tensor B_j[r,s,t] defines a map
    |r>_{qj}|0>_{q(j+1)} -> sum_{s,t} B_j[r,s,t]|s>_{qj}|t>_{q(j+1)}.
    The local Qiskit matrix row index is s + 2*t. The final MPS core
    gives the final single-qubit map |r> -> sum_s B[r,s,0]|s>.
    """
    if not cores:
        raise ValueError("Need at least one MPS core")
    n = len(cores)
    gates = []
    for j in range(n - 1):
        core = np.asarray(cores[j])
        if core.shape[0] > 2 or core.shape[2] > 2:
            raise ValueError("This layer requires bond dimension <= 2")
        # Embed outgoing bond < 2 in the |0> subspace of q(j+1).
        # Local row basis: bit(qj) + 2*bit(q(j+1)).
        w = np.zeros((4, core.shape[0]), dtype=core.dtype)
        for outgoing in range(core.shape[2]):
            for physical in range(2):
                w[physical + 2*outgoing, :] = core[:, physical, outgoing]
        gates.append(LocalGate((j, j + 1), _complete_isometry(w)))
    last = np.asarray(cores[-1])
    w_last = last[:, :, 0].T  # (physical=2, incoming_bond)
    gates.append(LocalGate((n - 1,), _complete_isometry(w_last)))
    return MPSLayer(tuple(gates), approximation_fidelity,
                    tuple(int(c.shape[2]) for c in cores[:-1]))


def apply_local_unitary(state: np.ndarray, matrix: np.ndarray,
                        qubits: tuple[int, ...], *, adjoint=False) -> np.ndarray:
    """Apply a 1q/2q matrix in Qiskit local little-endian basis ordering."""
    state = np.asarray(state)
    n = state.size.bit_length() - 1
    if state.ndim != 1 or state.size != (1 << n):
        raise ValueError("Expected a 2**n statevector")
    if len(qubits) not in (1, 2) or len(set(qubits)) != len(qubits) or any(q < 0 or q >= n for q in qubits):
        raise ValueError("Invalid gate qubits")
    u = np.asarray(matrix)
    k = len(qubits)
    if u.shape != (1 << k, 1 << k):
        raise ValueError("Unitary shape does not match qubit count")
    axes = [n - 1 - q for q in reversed(qubits)]
    perm = axes + [i for i in range(n) if i not in axes]
    v = state.reshape((2,) * n).transpose(perm).reshape(1 << k, -1)
    out = (u.conj().T if adjoint else u) @ v
    return out.reshape((2,) * n).transpose(np.argsort(perm)).reshape(-1)


def prepare_iterative_mps(target: np.ndarray, n_layers: int = 1, *,
                          max_bond: int = 2) -> IterativeMPSResult:
    """Greedy iterative MPS circuit approximation (paper Algorithm 2).

    No optimiser is called and gate locations are fixed by the MPS chain.
    The residual is updated with each layer's adjoint. The returned circuit
    applies the layers in *reverse* extraction order.
    """
    if not isinstance(n_layers, (int, np.integer)) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    if max_bond != 2:
        raise ValueError("Iterative two-qubit construction requires max_bond=2")
    psi = normalize_state(target)
    n = psi.size.bit_length() - 1
    residual = psi.copy()
    layers = []
    fidelities = []
    for _ in range(n_layers):
        compressed = decompose_state(residual, max_bond=2)
        layer = layer_from_mps(compressed.cores,
                               approximation_fidelity=compressed.fidelity)
        # For a chi=2 target this equals the contracted MPS to roundoff.
        # For a higher-chi target this uses a truncated and normalized MPS.
        residual = apply_layer_adjoint(residual, layer)
        f = float(np.abs(residual[0]) ** 2)
        layers.append(layer)
        fidelities.append(f)
    return IterativeMPSResult(tuple(layers), tuple(fidelities), n)


def apply_layer_adjoint(state: np.ndarray, layer: MPSLayer) -> np.ndarray:
    for gate in reversed(layer.gates):
        state = apply_local_unitary(state, gate.unitary, gate.qubits, adjoint=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative MPS preparation benchmark")
    ap.add_argument("--targets", default="aqce_results_2013/target_states.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--output", default="MPS/mps_fidelities.csv")
    ap.add_argument("--qasm", action="store_true", help="Save QASM2 circuit for the first target")
    args = ap.parse_args()
    if any(d < 1 for d in args.layers):
        ap.error("layers must all be positive")
    if args.max_states is not None and args.max_states < 1:
        ap.error("--max-states must be a positive integer")
    with np.load(args.targets) as d:
        targets = np.asarray(d["amplitudes"])
        dates = np.asarray(d["dates"]).astype(str)
    if args.max_states is not None:
        targets = targets[:args.max_states]
        dates = dates[:args.max_states]
    depth = max(args.layers)
    rows = []
    for idx, (date, psi) in enumerate(zip(dates, targets)):
        result = prepare_iterative_mps(psi, n_layers=depth)
        for layer_count in args.layers:
            f = result.fidelities[layer_count - 1]
            rows.append(dict(index=idx, date=date, layers=layer_count,
                             fidelity=f, infidelity=1 - f,
                             two_qubit_blocks=(result.n_qubits-1)*layer_count,
                             nominal_real_cx=2*(result.n_qubits-1)*layer_count))
        if args.qasm and idx == 0:
            circuit = result.to_qiskit(decompose_to_cx=True)
            from qiskit import qasm2
            from qiskit.quantum_info import Statevector
            compiled_f = abs(np.vdot(psi, Statevector.from_instruction(circuit).data)) ** 2
            if abs(compiled_f - result.fidelity) > 1e-8:
                raise AssertionError(f"Compiled Qiskit circuit changed fidelity: {compiled_f} vs {result.fidelity}")
            dest = Path(args.output).with_suffix(".qasm")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(qasm2.dumps(circuit))
            print("Saved first-state QASM:", dest)
            print("Compiled gate counts:", circuit.count_ops())
        if (idx+1) % 25 == 0 or idx+1 == len(targets):
            print(f"Processed {idx+1}/{len(targets)} targets", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} target/depth comparisons to {out}")
    for d in args.layers:
        fs = np.array([row["fidelity"] for row in rows if row["layers"] == d])
        print(f"D={d}: min={fs.min():.8f} mean={fs.mean():.8f} "
              f">=0.995: {int(np.sum(fs>=0.995))}/{len(fs)}")


if __name__ == "__main__":
    main()
