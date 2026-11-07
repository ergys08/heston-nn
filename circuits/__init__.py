"""Differentiable PyTorch PQCs for Heston amplitude state preparation.

The default 'yz' rotation is Ry(a)@Rz(b). 'rot' is Rz(a)@Ry(b)@Rz(c).
"""

from .brickwork import gate_counts as brickwork_gate_counts
from .brickwork import num_parameters as brickwork_num_parameters
from .brickwork import run_brickwork_circuit
from .chain import gate_counts as chain_gate_counts
from .chain import num_parameters as chain_num_parameters
from .chain import run_chain_circuit
from .ring import gate_counts as ring_gate_counts
from .ring import num_parameters as ring_num_parameters
from .ring import run_ring_circuit

__all__ = [
    "run_brickwork_circuit", "run_chain_circuit", "run_ring_circuit",
    "brickwork_num_parameters", "chain_num_parameters", "ring_num_parameters",
    "brickwork_gate_counts", "chain_gate_counts", "ring_gate_counts",
]
