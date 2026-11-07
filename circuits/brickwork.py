"""Standard brickwork Heston PQC, implemented directly in PyTorch.

A layer comprises even-bond CNOTs followed by odd-bond CNOTs. Each CNOT
is followed by a separate local rotation on both participating qubits,
matching the rotation-block count in Koike-Akino (2026), Table I.
"""

from __future__ import annotations

import torch

from ._core import (
    apply_cnot, apply_local_rotation, prepare_parameters, rotation_width,
    validate_dimensions, zero_state,
)


def gate_counts(num_qubits: int, num_layers: int, rotation: str = "yz") -> dict[str, int]:
    """Count CNOTs, local rotation blocks, and trainable rotation angles."""
    validate_dimensions(num_qubits, num_layers)
    width = rotation_width(rotation)
    cnots = (num_qubits - 1) * num_layers
    blocks = num_qubits + 2 * cnots
    return {"cnots": cnots, "rotation_blocks": blocks, "parameters": width * blocks}


def num_parameters(num_qubits: int, num_layers: int, rotation: str = "yz") -> int:
    return gate_counts(num_qubits, num_layers, rotation)["parameters"]


def run_brickwork_circuit(
    num_qubits: int, num_layers: int, parameters: torch.Tensor, rotation: str = "yz"
) -> torch.Tensor:
    """State preparation with autograd, returning (2**n,) or (batch, 2**n).

    Parameter order: initial rotations q=0..n-1, then per entangling
    layer even-bond and odd-bond CNOTs, with a fresh rotation on control
    and target after *each* CNOT. 'yz' means Ry(a)@Rz(b), right-to-left.
    """
    counts = gate_counts(num_qubits, num_layers, rotation)
    angles, single = prepare_parameters(parameters, counts["parameters"])
    width = rotation_width(rotation)
    state = zero_state(angles.shape[0], num_qubits, angles.device)
    offset = 0

    for qubit in range(num_qubits):
        state = apply_local_rotation(
            state, angles[:, offset:offset + width], qubit, num_qubits, rotation
        )
        offset += width

    for _ in range(num_layers):
        for start in (0, 1):
            for control in range(start, num_qubits - 1, 2):
                target = control + 1
                state = apply_cnot(state, control, target, num_qubits)
                for qubit in (control, target):
                    state = apply_local_rotation(
                        state, angles[:, offset:offset + width], qubit, num_qubits, rotation
                    )
                    offset += width

    return state[0] if single else state
