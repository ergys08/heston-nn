"""Unit tests for differentiable Heston PQC topologies (pytest)."""

import pytest
import torch

from circuits import (
    brickwork_gate_counts, chain_gate_counts, ring_gate_counts,
    run_brickwork_circuit, run_chain_circuit, run_ring_circuit,
)
from circuits._core import apply_cnot, local_rotation


@pytest.mark.parametrize("run,counts", [
    (run_brickwork_circuit, brickwork_gate_counts),
    (run_chain_circuit, chain_gate_counts),
    (run_ring_circuit, ring_gate_counts),
])
@pytest.mark.parametrize("rotation", ["yz", "rot"])
def test_normalization_batch_and_gradients(run, counts, rotation):
    n, layers = 4, 2
    p = counts(n, layers, rotation)["parameters"]
    theta = torch.randn(3, p, dtype=torch.float64, requires_grad=True)
    states = run(n, layers, theta, rotation)
    assert states.shape == (3, 2 ** n)
    torch.testing.assert_close(
        (states.conj() * states).sum(-1).real,
        torch.ones(3, dtype=torch.float64),
        rtol=1e-12, atol=1e-12,
    )
    target = torch.randn(3, 2 ** n, dtype=torch.complex128)
    loss = torch.abs((target.conj() * states).sum(-1)).square().sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().max() > 1e-10


@pytest.mark.parametrize("run,counts", [
    (run_brickwork_circuit, brickwork_gate_counts),
    (run_chain_circuit, chain_gate_counts),
    (run_ring_circuit, ring_gate_counts),
])
def test_zero_angles_return_zero_state(run, counts):
    n, layers = 4, 2
    theta = torch.zeros(counts(n, layers)["parameters"], dtype=torch.float64)
    output = run(n, layers, theta)
    expected = torch.zeros(2 ** n, dtype=torch.complex128)
    expected[0] = 1.0
    torch.testing.assert_close(output, expected)


def test_gate_counts_match_paper():
    assert brickwork_gate_counts(8, 18) == {
        "cnots": 126, "rotation_blocks": 260, "parameters": 520
    }
    assert chain_gate_counts(8, 18) == {
        "cnots": 126, "rotation_blocks": 152, "parameters": 304
    }
    assert ring_gate_counts(8, 16) == {
        "cnots": 128, "rotation_blocks": 136, "parameters": 272
    }


def test_little_endian_cnot_and_ring_edge():
    state = torch.zeros(1, 16, dtype=torch.complex128)
    state[0, 8] = 1
    result = apply_cnot(state, 3, 0, 4)
    assert result[0, 9] == 1
    assert torch.count_nonzero(result) == 1


def test_yz_matrix_order():
    a, b = 0.6, 1.1
    angles = torch.tensor([[a, b]], dtype=torch.float64, requires_grad=True)
    actual = local_rotation(angles, "yz")[0]
    aa = torch.tensor(a / 2, dtype=torch.float64)
    ry = torch.stack(
        (torch.stack((torch.cos(aa), -torch.sin(aa))),
         torch.stack((torch.sin(aa), torch.cos(aa))))
    ).to(torch.complex128)
    rz = torch.diag(torch.exp(1j * torch.tensor([-b / 2, b / 2])))
    torch.testing.assert_close(actual, ry @ rz, rtol=1e-7, atol=1e-7)


def test_invalid_parameter_shape():
    with pytest.raises(ValueError, match="Expected parameters"):
        run_chain_circuit(4, 2, torch.randn(10))
