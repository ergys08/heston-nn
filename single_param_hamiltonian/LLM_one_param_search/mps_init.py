"""Initialise Pauli-rotation angles from the iterative MPS circuit (MPS/).

The MPS staircase applies SO(4) gates on Qiskit pairs (j, j+1), j = 0..6
(least significant first) and one final single-qubit gate on Qiskit qubit 7
(paper qubit 1, the MSB).  Each real SO(4) gate is converted into six
rotation angles for the generator order a, b, YX, YZ, XY, ZY on the paper
pair (6-j, 7-j); the conversion is an exact nonlinear least-squares solve.

Resulting label order ("MPS order"): 8 leading Ry (zero), then for each
layer the blocks (6,7),(5,6),...,(0,1) followed by Y on paper qubit 0.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from MPS.circuit import prepare_iterative_mps  # noqa: E402
from paulis import odd_y_strings  # noqa: E402

ORDER = ["a", "b", "YX", "YZ", "XY", "ZY"]
_P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
_LOCAL = {tok: (-1j * np.kron(_P[tok[0] if tok not in "ab" else ("Y" if tok == "a" else "I")],
                              _P[tok[1] if tok not in "ab" else ("I" if tok == "a" else "Y")])).real for tok in ORDER}


def block_labels_pair(i, j):
    singles = {"a": odd_y_strings((i,)), "b": odd_y_strings((j,))}
    two = {l[i] + l[j]: l for l in odd_y_strings((i, j))}
    out = []
    for tok in ORDER:
        out += singles[tok] if tok in singles else [two[tok]]
    return out


def mps_order_labels(n_layers):
    labels = [l for q in range(8) for l in odd_y_strings((q,))]
    for _ in range(n_layers):
        for j in range(7):
            labels += block_labels_pair(6 - j, 7 - j)
        labels += odd_y_strings((0,))
    return labels


def so4_to_angles(u, rng, tries=20):
    u = np.asarray(u).real
    def resid(th):
        U = np.eye(4)
        for k, tok in enumerate(ORDER):
            U = expm(th[k] * _LOCAL[tok]) @ U
        return (U - u).ravel()
    best = None
    for t in range(tries):
        x0 = np.zeros(6) if t == 0 else rng.uniform(-np.pi, np.pi, 6)
        r = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or r.cost < best.cost:
            best = r
        if best.cost < 1e-20:
            break
    if best.cost > 1e-12:
        raise ArithmeticError(f"could not express SO(4) gate with six rotations (residual {best.cost:.2e})")
    return best.x


def so2_to_angle(u):
    u = np.asarray(u).real
    th = np.arctan2(u[1, 0], u[0, 0])
    if not np.allclose(expm(th * np.array([[0, -1], [1, 0]])), u, atol=1e-9):
        raise ArithmeticError("final gate is not a rotation")
    return th


def mps_angles(target, n_layers, seed=0):
    """Return (labels, theta) such that ProductAnsatz(labels).states(theta)
    reproduces the iterative MPS circuit state for `target`."""
    rng = np.random.default_rng(seed)
    res = prepare_iterative_mps(np.asarray(target, float), n_layers=n_layers)
    theta = [0.0] * 8
    Z7 = np.diag([1.0, 1.0, -1.0, -1.0])   # Z on Qiskit q7 in the (q6,q7) local basis
    for layer in reversed(res.layers):
        gates = [(g.qubits, np.asarray(g.unitary).real.copy()) for g in layer.gates]
        # A det=-1 final 1q gate R*Z is converted to a rotation by conjugating the
        # preceding (6,7) gate with Z: q7 is |0> before that gate, so Z acts trivially there.
        if np.linalg.det(gates[-1][1]) < 0:
            gates[-1] = (gates[-1][0], gates[-1][1] @ np.diag([1.0, -1.0]))
            gates[-2] = (gates[-2][0], Z7 @ gates[-2][1] @ Z7)
        for qubits, u in gates:
            if len(qubits) == 2:
                theta += list(so4_to_angles(u, rng))
            else:
                theta.append(so2_to_angle(u))
    labels = mps_order_labels(n_layers)
    assert len(labels) == len(theta)
    return labels, np.array(theta), res.fidelities


if __name__ == "__main__":
    from ansatz import ProductAnsatz, fit
    from states import load_states
    d, S, _ = load_states(2013)
    for L in (1, 2):
        labels, th, fids = mps_angles(S[0], L)
        phi = ProductAnsatz(labels).states(th[None, :])[0]
        print(f"layers {L}: MPS fidelity {fids[-1]:.7f}; reconstructed Pauli circuit fidelity {(phi @ S[0])**2:.7f}")
        r = fit(labels, S[[0]], variable=labels, init_var=th[None, :], restarts=1, maxiter=2000)
        print(f"   after L-BFGS polish from MPS init: {r.mean_fidelity:.7f}")
