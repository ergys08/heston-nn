"""Tests of SuperEncoder state-to-angle training on each new PQC."""

import pytest
import torch
import torch.nn as nn

from circuits import (
    brickwork_num_parameters, chain_num_parameters, ring_num_parameters,
    run_brickwork_circuit, run_chain_circuit, run_ring_circuit,
)
from loss import superencoder_loss
from mlp import SuperEncoderMLP
from train import train_superencoder


@pytest.mark.parametrize("topology,counts,run", [
    ("brickwork", brickwork_num_parameters, run_brickwork_circuit),
    ("chain", chain_num_parameters, run_chain_circuit),
    ("ring", ring_num_parameters, run_ring_circuit),
])
def test_mlp_targets_state_not_heston_parameters(topology, counts, run):
    torch.manual_seed(1)
    model = SuperEncoderMLP(4, 1, topology)
    assert isinstance(model.mlp[1], nn.Tanh)
    assert isinstance(model.mlp[-1], nn.Tanh)
    states = torch.randn(3, 16)
    angles = model(states)
    assert angles.shape == (3, counts(4, 1))
    assert torch.max(torch.abs(angles)) <= 1
    predictions = run(4, 1, angles)
    loss = superencoder_loss(states, predictions)
    loss.backward()
    assert model.mlp[0].weight.grad is not None
    assert torch.isfinite(model.mlp[0].weight.grad).all()
    assert torch.count_nonzero(model.mlp[0].weight.grad) > 0


@pytest.mark.parametrize("topology", ["brickwork", "chain", "ring"])
def test_end_to_end_training(topology):
    torch.manual_seed(3)
    states = torch.randn(3, 8, dtype=torch.float64)
    model, history = train_superencoder(
        training_states=states,
        num_qubits=3,
        num_layers=1,
        epochs=1,
        batch_size=2,
        learning_rate=3e-3,
        weight_decay=1e-5,
        topology=topology,
    )
    assert isinstance(model, SuperEncoderMLP)
    assert len(history) == 1 and torch.isfinite(torch.tensor(history)).all()
    assert 0 <= history[0] <= 1 + 1e-10


def test_batched_fidelity_is_average_of_individual_fidelities():
    targets = torch.tensor([[1., 0.], [0., 1.]])
    outputs = torch.tensor([[1., 0.], [1., 0.]])
    loss = superencoder_loss(targets, outputs)
    torch.testing.assert_close(loss, torch.tensor(.5, dtype=torch.float64))
    assert superencoder_loss(targets[0], outputs[0]).item() == pytest.approx(0.)


def test_loss_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="State dimensions"):
        superencoder_loss(torch.ones(4), torch.ones(1, 4))


def test_train_rejects_probability_not_amplitude_mismatch():
    with pytest.raises(ValueError, match="shape"):
        train_superencoder(torch.rand(4, 4), 3, 1, 1, 2, 3e-3, 1e-5)
