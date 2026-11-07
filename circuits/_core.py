"""Differentiable PyTorch primitives shared by the Heston PQC ansatze.

The simulator follows the little-endian convention from superencoder/circuit.py:
qubit 0 is the least significant computational-basis bit. No Qiskit dependency.
"""

from __future__ import annotations

import torch

ROTATION_PARAMETERS = {"yz": 2, "rot": 3}


def validate_dimensions(num_qubits: int, num_layers: int, minimum_qubits: int = 2) -> None:
    if not isinstance(num_qubits, int) or num_qubits < minimum_qubits:
        raise ValueError(f"num_qubits must be an integer >= {minimum_qubits}.")
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be a positive integer.")


def rotation_width(rotation: str) -> int:
    if rotation not in ROTATION_PARAMETERS:
        raise ValueError("rotation must be 'yz' or 'rot'.")
    return ROTATION_PARAMETERS[rotation]


def prepare_parameters(parameters: torch.Tensor, expected: int) -> tuple[torch.Tensor, bool]:
    """Accept (P,) or (batch, P); keep autograd and the input device."""
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor to preserve autograd.")
    if not torch.is_floating_point(parameters):
        raise TypeError("parameters must have a real floating-point dtype.")
    if parameters.ndim not in (1, 2) or parameters.shape[-1] != expected:
        raise ValueError(
            f"Expected parameters of shape ({expected},) or (batch, {expected}); "
            f"received {tuple(parameters.shape)}."
        )
    single = parameters.ndim == 1
    batch = parameters.reshape(-1, expected).to(torch.float64)
    return batch, single


def zero_state(batch_size: int, num_qubits: int, device: torch.device) -> torch.Tensor:
    state = torch.zeros(batch_size, 1 << num_qubits, dtype=torch.complex128, device=device)
    state[:, 0] = 1.0
    return state


def ry_gate(theta: torch.Tensor) -> torch.Tensor:
    """Batched 2x2 Ry matrices, differentiable in theta."""
    cosine = torch.cos(theta / 2)
    sine = torch.sin(theta / 2)
    return torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    ).to(torch.complex128)


def rz_gate(theta: torch.Tensor) -> torch.Tensor:
    """Batched 2x2 Rz matrices, differentiable in theta."""
    minus = torch.exp((-0.5j) * theta)
    plus = torch.exp((0.5j) * theta)
    zeros = torch.zeros_like(minus)
    return torch.stack(
        (
            torch.stack((minus, zeros), dim=-1),
            torch.stack((zeros, plus), dim=-1),
        ),
        dim=-2,
    )


def local_rotation(angles: torch.Tensor, rotation: str) -> torch.Tensor:
    """Return Ry(a)@Rz(b) ('yz') or Rz(a)@Ry(b)@Rz(c) ('rot').

    Matrix products follow the conventional right-to-left application order.
    """
    if rotation == "yz":
        return ry_gate(angles[:, 0]) @ rz_gate(angles[:, 1])
    if rotation == "rot":
        return rz_gate(angles[:, 0]) @ ry_gate(angles[:, 1]) @ rz_gate(angles[:, 2])
    raise ValueError("rotation must be 'yz' or 'rot'.")


def apply_single_qubit_gate(
    state: torch.Tensor, gate: torch.Tensor, qubit: int, num_qubits: int
) -> torch.Tensor:
    """Apply batched 2x2 gates without constructing a 2**n by 2**n matrix."""
    if not 0 <= qubit < num_qubits:
        raise ValueError("qubit index is out of range.")
    batch_size = state.shape[0]
    axis = num_qubits - qubit  # leading batch axis, then big-endian tensor axes
    tensor = state.reshape(batch_size, *([2] * num_qubits)).movedim(axis, 1)
    shape = tensor.shape
    tensor = torch.matmul(gate, tensor.reshape(batch_size, 2, -1))
    return tensor.reshape(shape).movedim(1, axis).reshape(batch_size, -1)


def apply_local_rotation(
    state: torch.Tensor, angles: torch.Tensor, qubit: int,
    num_qubits: int, rotation: str
) -> torch.Tensor:
    return apply_single_qubit_gate(state, local_rotation(angles, rotation), qubit, num_qubits)


def apply_cnot(
    state: torch.Tensor, control: int, target: int, num_qubits: int
) -> torch.Tensor:
    """Apply CNOT via a differentiable computational-basis permutation."""
    if not (0 <= control < num_qubits and 0 <= target < num_qubits) or control == target:
        raise ValueError("control and target must be distinct valid qubit indices.")
    indices = torch.arange(1 << num_qubits, device=state.device)
    flip = (((indices >> control) & 1) * (1 << target))
    return state.index_select(-1, indices ^ flip)
