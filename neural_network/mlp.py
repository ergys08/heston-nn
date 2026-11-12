"""SuperEncoder MLP: target state amplitudes -> PQC rotation angles.

This deliberately retains the input representation and both Tanh activations
of superencoder/mlp.py. Only the output dimension is adapted to each PQC.
It does NOT learn Heston model parameters -> circuit angles.
"""

import torch
import torch.nn as nn

from circuits import (
    brickwork_num_parameters,
    chain_num_parameters,
    ring_num_parameters,
)

_PARAMETER_COUNTS = {
    "brickwork": brickwork_num_parameters,
    "chain": chain_num_parameters,
    "ring": ring_num_parameters,
}


class SuperEncoderMLP(nn.Module):
    """Predict circuit parameters from the target state's real amplitudes.

    Args:
        num_qubits: Number of qubits in the amplitude encoding.
        num_layers: Entangling layers in the selected PQC.
        topology: 'brickwork', 'chain', or 'ring'.
        rotation: 'yz' (Ry@Rz, 2 angles) or 'rot' (Euler, 3 angles).

    Input shape: (2**num_qubits,) or (batch, 2**num_qubits).
    Output shape: (num_parameters,) or (batch, num_parameters).

    Preserves the original SuperEncoder architecture:
        Linear(2**n, 512) -> Tanh -> Linear(512, P) -> Tanh.
    The final Tanh restricts raw angles to (-1, 1) radians, exactly as
    in the original repository. This is not an unrestricted angle head.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        topology: str = "brickwork",
        rotation: str = "yz",
    ) -> None:
        super().__init__()
        if topology not in _PARAMETER_COUNTS:
            raise ValueError("topology must be 'brickwork', 'chain', or 'ring'.")
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.topology = topology
        self.rotation = rotation
        self.num_parameters = _PARAMETER_COUNTS[topology](num_qubits, num_layers, rotation)
        self.mlp = nn.Sequential(
            nn.Linear(2 ** num_qubits, 512),
            nn.Tanh(),
            nn.Linear(512, self.num_parameters),
            nn.Tanh(),
        )

    def forward(self, target_state: torch.Tensor) -> torch.Tensor:
        """Predict PQC parameters from a real-valued target statevector."""
        return self.mlp(target_state)
