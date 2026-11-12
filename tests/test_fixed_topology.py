"""Tests for the fixed-topology two-qubit block optimiser (pytest)."""

import json
from pathlib import Path

import numpy as np
import pytest

from circuits.fixed_topology import (
    apply_block, canonical_topology, fidelities, greedy_disentangle,
    haar_unitaries, is_unitary, optimise, pairs_from_qulacs, prepare, sweep,
    swap_matrix_qubits,
)

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = [(0, 1), (2, 5), (1, 7), (3, 4), (6, 7), (0, 3), (4, 6), (2, 7)]


def dense_block(u, a, b, n=8):
    """Full 2**n matrix of a block on (a, b), built element by element."""
    dim = 1 << n
    full = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        s = 2 * ((col >> a) & 1) + ((col >> b) & 1)
        for r in range(4):
            row = col & ~(1 << a) & ~(1 << b)
            row |= ((r >> 1) & 1) << a | (r & 1) << b
            full[row, col] += u[r, s]
    return full


def test_apply_block_matches_dense_matrix():
    rng = np.random.default_rng(1)
    u = haar_unitaries(rng, (1,))[0]
    psi = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    for a, b in [(0, 1), (5, 2), (7, 0)]:
        expected = dense_block(u, a, b) @ psi
        got = apply_block(psi[None], u[None], (a, b))[0]
        np.testing.assert_allclose(got, expected, atol=1e-12)
        back = apply_block(got[None], u[None], (a, b), adjoint=True)[0]
        np.testing.assert_allclose(back, psi, atol=1e-12)


def test_swap_matrix_qubits_relabels_pair():
    rng = np.random.default_rng(2)
    u = haar_unitaries(rng, (1,))[0]
    psi = rng.standard_normal((1, 256)).astype(complex)
    np.testing.assert_allclose(apply_block(psi, u[None], (2, 6)),
                               apply_block(psi, swap_matrix_qubits(u)[None], (6, 2)),
                               atol=1e-12)


def test_matches_qulacs_convention():
    qulacs = pytest.importorskip("qulacs")
    from qulacs.gate import DenseMatrix
    rng = np.random.default_rng(3)
    u = haar_unitaries(rng, (len(TOPOLOGY),))
    state = qulacs.QuantumState(8)
    for (a, b), m in zip(TOPOLOGY, u):
        DenseMatrix([b, a], m).update_quantum_state(state)
    np.testing.assert_allclose(prepare(TOPOLOGY, u[None])[0], state.get_vector(), atol=1e-12)


def test_initialisers_and_optimised_blocks_are_unitary():
    rng = np.random.default_rng(4)
    targets = np.abs(rng.standard_normal((3, 256)))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    assert is_unitary(haar_unitaries(rng, (5, 3)))
    assert is_unitary(greedy_disentangle(targets.astype(complex), TOPOLOGY))
    res = optimise(targets, TOPOLOGY, restarts=3, sweeps=20)
    assert is_unitary(res.unitaries)
    np.testing.assert_allclose(fidelities(targets.astype(complex), TOPOLOGY, res.unitaries),
                               res.fidelity, atol=1e-12)
    assert np.all((res.fidelity >= 0) & (res.fidelity <= 1 + 1e-12))


def test_sweeps_never_decrease_fidelity():
    rng = np.random.default_rng(5)
    targets = rng.standard_normal((4, 256)) + 1j * rng.standard_normal((4, 256))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    u = haar_unitaries(rng, (4, len(TOPOLOGY)))
    previous = fidelities(targets, TOPOLOGY, u)
    for _ in range(10):
        current = sweep(targets, TOPOLOGY, u)
        np.testing.assert_allclose(current, fidelities(targets, TOPOLOGY, u), atol=1e-12)
        assert np.all(current >= previous - 1e-12)
        previous = current


def test_recovers_state_generated_by_same_topology():
    rng = np.random.default_rng(6)
    topology = [(0, 1), (1, 2), (2, 3)]
    true_u = haar_unitaries(rng, (1, 3))
    target = prepare(topology, true_u, n=4)
    res = optimise(target, topology, restarts=4, sweeps=300, n=4, seed=7)
    assert res.fidelity[0] > 1 - 1e-9


def test_canonical_topology():
    assert canonical_topology([(3, 1), [0, 2]]) == [(1, 3), (0, 2)]
    with pytest.raises(ValueError):
        canonical_topology([(2, 2)])


@pytest.mark.parametrize("folder", ["aqce_results_2013", "aqce_results_2013_M20"])
def test_reproduces_saved_aqce_fidelities(folder):
    path = ROOT / folder
    if not (path / "target_states.npz").exists():
        pytest.skip("saved AQCE results unavailable")
    targets = np.load(path / "target_states.npz")["amplitudes"]
    for i in (0, 57, 123, 249):
        rec = json.loads((path / "states" / f"{i:03d}.json").read_text())
        saved = np.load(path / "unitaries" / f"{i:03d}.npz")
        topology = pairs_from_qulacs(saved["pairs"])
        f = fidelities(targets[i:i + 1].astype(complex), topology, saved["matrices"][None])
        assert abs(f[0] - rec["fidelity"]) < 1e-10
        u = saved["matrices"].copy()
        for m, (a, b) in enumerate(topology):
            if a > b:
                u[m] = swap_matrix_qubits(u[m])
        refined = optimise(targets[i:i + 1], topology, restarts=1, sweeps=5,
                           warm_start=u[None])
        assert refined.fidelity[0] >= rec["fidelity"] - 1e-12
