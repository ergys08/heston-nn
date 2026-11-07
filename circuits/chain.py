"""Sequential nearest-neighbour chain Heston PQC (PyTorch autograd)."""

from __future__ import annotations

import torch

from ._core import (
    apply_cnot, apply_local_rotation, prepare_parameters, rotation_width,
    validate_dimensions, zero_state,
)


def gate_counts(num_qubits: int, num_layers: int, rotation: str = "yz") -> dict[str, int]:
    """One forward CNOT chain per layer, with full rotation banks between chains."""
    validate_dimensions(num_qubits, num_layers)
    width = rotation_width(rotation)
    cnots = (num_qubits - 1) * num_layers
    blocks = num_qubits * (num_layers + 1)
    return {"cnots": cnots, "rotation_blocks": blocks, "parameters": width * blocks}


def num_parameters(num_qubits: int, num_layers: int, rotation: str = "yz") -> int:
    return gate_counts(num_qubits, num_layers, rotation)["parameters"]


def run_chain_circuit(
    num_qubits: int, num_layers: int, parameters: torch.Tensor, rotation: str = "yz"
) -> torch.Tensor:
    """State preparation via sequential CNOT(0,1), ..., CNOT(n-2,n-1).

    Initial rotations and one full bank following each CNOT chain.
    Parameters are flat, ordered by bank then qubit. Accepts (P,) or
    (batch, P), and preserves PyTorch gradients.
    """
    counts = gate_counts(num_qubits, num_layers, rotation)
    angles, single = prepare_parameters(parameters, counts["parameters"])
    width = rotation_width(rotation)
    state = zero_state(angles.shape[0], num_qubits, angles.device)
    offset = 0

    for layer in range(num_layers + 1):
        if layer:
            for control in range(num_qubits - 1):
                state = apply_cnot(state, control, control + 1, num_qubits)
        for qubit in range(num_qubits):
            state = apply_local_rotation(
                state, angles[:, offset:offset + width], qubit, num_qubits, rotation
            )
            offset += width

    return state[0] if single else state
