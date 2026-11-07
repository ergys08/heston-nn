"""End-to-end SuperEncoder training on target quantum state amplitudes.

Retains superencoder/train.py's target-state -> MLP -> PQC -> fidelity loss
workflow, per-target loss averaging, shuffled DataLoader, Adam, and
(model, history) return signature. The PQC is selectable, and the MLP's
output width is derived from that PQC's angle count.

Reference hyperparameters in the original repository: 10 epochs, batch size
32 for 4 qubits or 64 for 6/8 qubits, Adam lr=3e-3, weight_decay=1e-5.
These are reference settings, not forced defaults.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from circuits import run_brickwork_circuit, run_chain_circuit, run_ring_circuit
from loss import superencoder_loss
from mlp import SuperEncoderMLP

_CIRCUITS = {
    "brickwork": run_brickwork_circuit,
    "chain": run_chain_circuit,
    "ring": run_ring_circuit,
}


def train_superencoder(
    training_states: torch.Tensor,
    num_qubits: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device = "cpu",
    topology: str = "brickwork",
    rotation: str = "yz",
) -> tuple[SuperEncoderMLP, list[float]]:
    """Train a state-to-angle SuperEncoder using the selected PyTorch PQC.

    training_states: (num_states, 2**num_qubits), **real state amplitudes**.
        Heston binned probability masses should be square-rooted before
        passing here: psi_j = sqrt(p_j), not p_j itself. Training-state
        order is not interpreted as chronology by this function.
    num_layers: entangling-layer count (replaces old num_blocks argument).
    topology: 'brickwork', 'chain', or 'ring'.
    rotation: 'yz' or 'rot'.

    Returns the trained MLP and per-epoch mean of minibatch fidelity losses.
    The statevector simulations remain in PyTorch's autograd graph.
    """
    if topology not in _CIRCUITS:
        raise ValueError("topology must be 'brickwork', 'chain', or 'ring'.")
    if not isinstance(training_states, torch.Tensor):
        raise TypeError("training_states must be a torch.Tensor.")
    if training_states.ndim != 2 or training_states.shape[1] != 2 ** num_qubits:
        raise ValueError("training_states must have shape (num_states, 2**num_qubits).")
    if training_states.shape[0] == 0:
        raise ValueError("training_states cannot be empty.")
    if not training_states.is_floating_point():
        raise TypeError("SuperEncoderMLP expects real floating-point state amplitudes.")
    if not torch.isfinite(training_states).all():
        raise ValueError("training_states must contain only finite values.")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Require learning_rate > 0 and weight_decay >= 0.")

    device = torch.device(device)
    training_states = training_states.to(device)
    norms = torch.linalg.vector_norm(training_states, dim=1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError("Training data contains a zero statevector.")
    training_states = training_states / norms

    dataset = TensorDataset(training_states)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SuperEncoderMLP(
        num_qubits=num_qubits,
        num_layers=num_layers,
        topology=topology,
        rotation=rotation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    run_circuit = _CIRCUITS[topology]
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            losses = []
            for target_state in batch:
                # Original SuperEncoder: MLP consumes the *target state*,
                # not Heston calibration parameters or separately fitted angles.
                parameters = model(target_state.to(dtype=torch.float32))
                predicted_state = run_circuit(
                    num_qubits=num_qubits,
                    num_layers=num_layers,
                    parameters=parameters,
                    rotation=rotation,
                )
                losses.append(superencoder_loss(target_state, predicted_state))

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            num_batches += 1

        mean_epoch_loss = epoch_loss / num_batches
        history.append(mean_epoch_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {mean_epoch_loss:.6f} | "
            f"Fidelity: {1.0 - mean_epoch_loss:.6f}"
        )

    return model, history
