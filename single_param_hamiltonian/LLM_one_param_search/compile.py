"""Compile a product of Pauli rotations into CNOT + single-qubit gates and
count CNOTs honestly with Qiskit.

Three counts are reported for an ordered list of Pauli labels:
  cx_ladder : naive CNOT ladder, 2*(weight-1) per term, no cancellation
  cx_opt1   : Qiskit optimisation level 1 (adjacent CNOT cancellation only,
              every rotation angle remains an explicit Rz)
  cx_opt3   : Qiskit optimisation level 3 (two-qubit block consolidation and
              KAK resynthesis; the standard "actual compiled" count)
Qubit ordering: paper qubit 1 (MSB) = Qiskit qubit 7, so the paper label
'IIYZIIII' is passed to Qiskit unchanged (Qiskit labels are little-endian).
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from paulis import normalise_label, weight

BASIS = ["rz", "sx", "x", "cx"]


def _qiskit():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    return QuantumCircuit, transpile, PauliEvolutionGate, SparsePauliOp, Statevector


def build_circuit(labels, thetas):
    """prod_k exp(-i theta_k P_k)|0>, first label applied first."""
    QuantumCircuit, _, PauliEvolutionGate, SparsePauliOp, _ = _qiskit()
    qc = QuantumCircuit(8)
    for label, th in zip(labels, thetas):
        label = normalise_label(label)
        if weight(label) == 1:
            q = 7 - label.index("Y")  # weight-1 odd-Y string is a single Y
            qc.ry(2 * float(th), q)
        else:
            qc.append(PauliEvolutionGate(SparsePauliOp(label), time=float(th)), range(8))
    return qc


def compiled_statevector(qc) -> np.ndarray:
    _, _, _, _, Statevector = _qiskit()
    return np.asarray(Statevector(qc).data)


@lru_cache(maxsize=None)
def _cx_counts_cached(labels_key, theta_key):
    _, transpile, _, _, Statevector = _qiskit()
    labels = list(labels_key); thetas = list(theta_key)
    qc = build_circuit(labels, thetas)
    ref = compiled_statevector(qc)
    out = {"cx_ladder": int(sum(2 * (weight(l) - 1) for l in labels))}
    for level in (1, 3):
        tq = transpile(qc, basis_gates=BASIS, optimization_level=level, seed_transpiler=0)
        sv = compiled_statevector(tq)
        if abs(np.vdot(ref, sv)) ** 2 < 1 - 1e-9:
            raise AssertionError("transpiled circuit does not reproduce the statevector")
        out[f"cx_opt{level}"] = int(tq.count_ops().get("cx", 0))
        out[f"depth_opt{level}"] = int(tq.depth())
    return out


def cx_counts(labels, thetas=None) -> dict:
    """CNOT counts for the ordered product.  Angles only matter through
    accidental simplifications, so generic (random) angles are used unless
    explicit ones are supplied."""
    labels = tuple(normalise_label(l) for l in labels)
    if thetas is None:
        rng = np.random.default_rng(12345)
        thetas = tuple(float(x) for x in rng.uniform(0.3, 1.2, len(labels)))
    else:
        thetas = tuple(float(x) for x in thetas)
    return _cx_counts_cached(labels, thetas)


def verify_against_simulator(labels, thetas, tol=1e-10) -> float:
    """Return fidelity between Qiskit statevector and our real simulator."""
    from ansatz import ProductAnsatz
    phi = ProductAnsatz(labels).states(np.asarray(thetas)[None, :])[0]
    sv = compiled_statevector(build_circuit(labels, thetas))
    f = abs(np.vdot(sv, phi)) ** 2
    if f < 1 - tol:
        raise AssertionError(f"simulator/Qiskit mismatch: fidelity {f}")
    return float(f)


if __name__ == "__main__":
    import time
    from paulis import pair_block_strings
    rng = np.random.default_rng(0)
    labels = ["IIYIIIII", "IIYZIIII", "IZYIIIII", "YXIIIIII", "IIIIYZZI", "IIIIIIIY", "IYIIIIII", "YIIIIIIZ"]
    th = rng.normal(size=len(labels))
    print("simulator vs qiskit fidelity:", verify_against_simulator(labels, th))
    t = time.time(); print(labels, cx_counts(labels), f"{time.time()-t:.2f}s")
    chain = sum([pair_block_strings(i, i + 1) for i in range(7)], [])
    t = time.time(); print("chain 7 x so(4) blocks:", cx_counts(chain), f"{time.time()-t:.2f}s")
    one_block = pair_block_strings(2, 3)
    print("one so(4) block:", cx_counts(one_block))
    print("single YZ rotation:", cx_counts(["IIYZIIII"]))
    print("YZ + YX same pair:", cx_counts(["IIYZIIII", "IIYXIIII"]))
    print("3-local YZZ:", cx_counts(["IIYZZIII"]))
    print("YZ, YZZ, YIZ sharing root:", cx_counts(["IIYZIIII", "IIYZZIII", "IIYIZIII"]))
