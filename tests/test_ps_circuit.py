"""Tests for PS layouts and compatibility with the Heston 3-CX block template."""

from collections import Counter
import numpy as np
import pytest

from MPS.ps_circuit import build_ps_layout, ps_topology, build_ps_circuit


@pytest.mark.parametrize("n", [5, 8, 9, 12])
def test_brickwall_limit(n):
    for M in (1, 2, 3):
        result = build_ps_layout(n, M=M, l=2, q=1)
        assert len(result.pairs) == M * (n - 1)
        assert result.two_qubit_depth == 2 * M
        assert Counter(result.pairs) == Counter(
            {(i, i+1): M for i in range(n-1)})
        assert set(result.layers[0]) == {
            (i, i+1) for i in range(0, n-1, 2)}


@pytest.mark.parametrize("n", [5, 8, 9, 12])
def test_sequential_limit(n):
    for M in (1, 2):
        result = build_ps_layout(n, M=M, l=n-1, q=1)
        assert len(result.pairs) == M * (n - 1)
        assert Counter(result.pairs) == Counter(
            {(i, i+1): M for i in range(n-1)})
        if M == 1:
            assert result.pairs == tuple((i, i+1) for i in range(n-1))
            assert result.two_qubit_depth == n-1


@pytest.mark.parametrize("n,M,l,q", [
    (8,1,3,2), (8,2,3,2), (12,2,4,3), (9,1,2,2)])
def test_overlap_and_conflict_free(n, M, l, q):
    result = build_ps_layout(n,M,l,q)
    assert result.n_blocks >= M*(n-1)
    assert result.n_cx == 3*result.n_blocks
    assert result.n_angles == 24*result.n_blocks
    for layer in result.layers:
        wires = [w for a,b in layer for w in (a,b)]
        assert len(wires) == len(set(wires))
    assert all(b == a+1 and 0 <= a < b < n for a,b in result.pairs)
    assert ps_topology(n,M,l,q) == list(result.pairs)
    assert Counter(result.pairs)[(l,l+1)] > M


@pytest.mark.parametrize("args", [
    (8,0,2,1), (8,1,1,1), (8,1,8,1),
    (8,1,3,0), (8,1,3,4), (2,1,2,1)])
def test_invalid_parameters(args):
    with pytest.raises(ValueError):
        build_ps_layout(*args)


def test_qiskit_gate_template_and_numeric_consistency():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
    from circuits.fixed_topology import prepare

    n, M, l, q = (5, 1, 3, 2)
    layout = build_ps_layout(n,M,l,q)
    circ, params = build_ps_circuit(n,M,l,q)
    ops = circ.count_ops()
    assert ops.get("cx",0) == 3*layout.n_blocks
    assert ops.get("rz",0) == 16*layout.n_blocks
    assert ops.get("ry",0) == 8*layout.n_blocks
    assert len(params) == 24*layout.n_blocks

    rng = np.random.default_rng(173)
    angles = rng.normal(size=(layout.n_blocks,4,2,3))
    bound, params2 = build_ps_circuit(n,M,l,q,angles=angles)
    assert [p.name for p in params] == [p.name for p in params2]
    via_symbols = circ.assign_parameters(dict(zip(params,angles.ravel())))
    np.testing.assert_allclose(
        Statevector.from_instruction(bound).data,
        Statevector.from_instruction(via_symbols).data,
        atol=1e-11,
    )

    # Validate local gate convention against existing NumPy simulator.
    mats = []
    for block in angles:
        local = QuantumCircuit(2)
        for slot in range(4):
            for wire in range(2):
                theta, phi, lam = block[slot,wire]
                local.rz(lam,wire)
                local.ry(theta,wire)
                local.rz(phi,wire)
            if slot < 3:
                local.cx(0,1)
        mats.append(Operator(local).data)
    simulated = prepare(layout.pairs,np.asarray(mats)[None,...],n=n)[0]
    np.testing.assert_allclose(
        simulated, Statevector.from_instruction(bound).data, atol=1e-11)
