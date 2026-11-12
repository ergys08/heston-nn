"""Numerical checks on decomposition, circuit ordering, and iterative residuals."""
import numpy as np
import pytest

from MPS.mps_decomposition import decompose_state, mps_to_statevector
from MPS.circuit import (prepare_iterative_mps, apply_local_unitary,
                         layer_from_mps, apply_layer_adjoint)


def assert_up_to_phase(actual, expected, tol=1e-10):
    f = abs(np.vdot(actual, expected)) ** 2 / (np.vdot(actual,actual).real * np.vdot(expected,expected).real)
    assert f > 1 - tol, f


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_bond2_mps_is_exactly_prepared(n):
    rng = np.random.default_rng(71 + n)
    cores = []
    right = 1
    for site in range(n-1, -1, -1):
        left = 1 if site == 0 else 2
        a = rng.normal(size=(2*right, left))
        q, _ = np.linalg.qr(a)
        cores.insert(0, q.T.reshape(left,2,right))
        right = left
    target = mps_to_statevector(cores)
    target /= np.linalg.norm(target)
    result = prepare_iterative_mps(target, 1)
    assert result.fidelity > 1-1e-10
    assert_up_to_phase(result.prepare_statevector(), target)
    for gate in result.layers[0].gates:
        u = gate.unitary
        np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-10)
        if not np.iscomplexobj(target) and len(gate.qubits) == 2:
            assert np.linalg.det(u) > 0.999999


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_random_complex_mps_reconstruction(n):
    rng = np.random.default_rng(n)
    s = rng.normal(size=2**n) + 1j*rng.normal(size=2**n)
    s /= np.linalg.norm(s)
    result = decompose_state(s, max_bond=2**n)
    assert result.fidelity > 1 - 1e-12
    assert_up_to_phase(result.to_statevector(), s)


def test_rank2_truncation_reconstruction_matches_layer():
    rng = np.random.default_rng(34)
    s = rng.normal(size=256)
    s /= np.linalg.norm(s)
    mps = decompose_state(s, max_bond=2)
    layer = layer_from_mps(mps.cores)
    output = np.zeros(256)
    output[0] = 1
    for gate in layer.gates:
        output = apply_local_unitary(output, gate.unitary, gate.qubits)
    assert_up_to_phase(output, mps.to_statevector())
    assert abs(abs(np.vdot(output, s))**2 - mps.fidelity) < 1e-10
    residual = apply_layer_adjoint(s, layer)
    assert abs(abs(residual[0])**2 - mps.fidelity) < 1e-10


def test_iterative_order_and_fidelity_consistency():
    rng = np.random.default_rng(121)
    s = rng.normal(size=64) + 1j*rng.normal(size=64)
    s /= np.linalg.norm(s)
    result = prepare_iterative_mps(s, 3)
    for d in range(1, 4):
        prefix = type(result)(result.layers[:d],result.fidelities[:d],result.n_qubits)
        output = prefix.prepare_statevector()
        assert abs(abs(np.vdot(output,s))**2 - result.fidelities[d-1]) < 1e-10


def test_qiskit_bit_convention_manual_cnot():
    s = np.zeros(8)
    s[1] = 1
    u = np.eye(4)[:, [0,3,2,1]]
    actual = apply_local_unitary(s,u,(0,2))
    assert np.argmax(abs(actual)) == 5


def test_bad_input():
    with pytest.raises(ValueError):
        prepare_iterative_mps(np.zeros(8),1)
    with pytest.raises(ValueError):
        prepare_iterative_mps(np.ones(8),0)
    with pytest.raises(ValueError):
        decompose_state(np.ones(7))


def test_narrow_real_state_handles_rank_one_bonds():
    x = np.arange(256, dtype=float)
    p = np.exp(-0.5*((x-128)/3)**2)
    psi = np.sqrt(p/p.sum())
    result = prepare_iterative_mps(psi, 4)
    assert all(0 <= f <= 1+1e-12 for f in result.fidelities)
    assert result.fidelities[-1] > 0.999
    assert abs(abs(np.vdot(psi,result.prepare_statevector()))**2 - result.fidelity)<1e-10


def test_optional_qiskit_output_matches_numpy():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(11)
    target = rng.normal(size=32)
    target /= np.linalg.norm(target)
    result = prepare_iterative_mps(target, 2)
    circuit = result.to_qiskit()
    np.testing.assert_allclose(Statevector.from_instruction(circuit).data,
                               result.prepare_statevector(), atol=1e-10)
    compiled = result.to_qiskit(decompose_to_cx=True)
    assert abs(abs(np.vdot(target,Statevector.from_instruction(compiled).data))**2
               - result.fidelity)<1e-8
