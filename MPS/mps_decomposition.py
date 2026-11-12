"""Dense-state to right-canonical MPS decomposition and bond truncation.

Implements the SVD compression used in Algorithm 2 of:
Iaconis, Johri & Zhu, *Quantum state preparation of normal distributions
using matrix product states*, npj Quantum Information 10, 15 (2024).
https://doi.org/10.1038/s41534-024-00805-0

Conventions: amplitudes are indexed as in Qiskit / the Heston repo:
q0 is the least-significant bit of the integer basis index. MPS cores are
ordered in the sequence q0, q1, ..., q(n-1), and have shapes
(left_bond, physical=2, right_bond).

This implementation begins with a full 2**n statevector; the paper can
instead work with an MPS input, avoiding exponential classical storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MPSDecomposition:
    """Normalized right-canonical MPS obtained by sequential SVDs."""

    cores: tuple[np.ndarray, ...]
    retained_norm: float  # norm of unnormalized truncated state
    discarded_weights: tuple[float, ...]  # right-to-left SVD discarded weights
    bond_dimensions: tuple[int, ...]  # length n-1
    fidelity: float  # squared overlap with normalized input

    def to_statevector(self) -> np.ndarray:
        return mps_to_statevector(self.cores)


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Validate and normalize a (2**n,) real/complex pure statevector."""
    psi = np.asarray(state)
    if psi.ndim != 1 or psi.size < 2 or psi.size & (psi.size - 1):
        raise ValueError("state must be a one-dimensional 2**n vector (n >= 1)")
    if not np.all(np.isfinite(psi)):
        raise ValueError("state must have finite complex amplitudes")
    psi = psi.astype(np.complex128 if np.iscomplexobj(psi) else np.float64)
    norm = float(np.linalg.norm(psi))
    if norm == 0:
        raise ValueError("cannot decompose the zero state")
    return psi / norm


def statevector_to_tensor(state: np.ndarray) -> np.ndarray:
    """Physical axes q0..q(n-1), not NumPy's default q(n-1)..q0."""
    psi = normalize_state(state)
    n = psi.size.bit_length() - 1
    return psi.reshape((2,) * n).transpose(tuple(reversed(range(n))))


def tensor_to_statevector(tensor: np.ndarray) -> np.ndarray:
    """Invert statevector_to_tensor without normalizing the tensor."""
    tensor = np.asarray(tensor)
    if tensor.ndim < 1 or any(d != 2 for d in tensor.shape):
        raise ValueError("tensor must have shape (2,) * n")
    return tensor.transpose(tuple(reversed(range(tensor.ndim)))).reshape(-1)


def mps_to_statevector(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Contract physical-ordered MPS cores to a Qiskit-order statevector."""
    if not cores:
        raise ValueError("MPS must have at least one tensor")
    if np.asarray(cores[0]).shape[0] != 1 or np.asarray(cores[-1]).shape[-1] != 1:
        raise ValueError("MPS boundary bonds must be dimension one")
    tensor = np.asarray(cores[0])[0]
    for core in cores[1:]:
        core = np.asarray(core)
        if core.ndim != 3 or core.shape[1] != 2 or tensor.shape[-1] != core.shape[0]:
            raise ValueError("Incompatible MPS bonds")
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    return tensor_to_statevector(tensor[..., 0])


def decompose_state(state: np.ndarray, max_bond: int = 2, *,
                    rank_tol: float = 1e-13) -> MPSDecomposition:
    """Right-to-left TT-SVD with bond dimension <= max_bond.

    For site j>0, A[j].reshape(chi_left, 2*chi_right) has orthonormal
    *rows*. This right-canonical form provides a 1-to-2-qubit isometry for
    constructing the nearest-neighbour preparation layer.

    Each truncated MPS is normalized before being used as a circuit target.
    The decomposition is best understood as the paper's SVD compression,
    not as a globally optimal fixed-rank MPS approximation.
    """
    if max_bond < 1 or not isinstance(max_bond, (int, np.integer)):
        raise ValueError("max_bond must be a positive integer")
    if not 0 <= rank_tol < 1:
        raise ValueError("rank_tol must be in [0, 1)")
    psi = normalize_state(state)
    n = psi.size.bit_length() - 1
    residual = statevector_to_tensor(psi)
    cores: list[np.ndarray | None] = [None] * n
    discarded = [0.0] * (n - 1)
    right_rank = 1
    for site in range(n - 1, 0, -1):
        matrix = residual.reshape(-1, 2 * right_rank)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        numerical_rank = max(1, int(np.count_nonzero(singular > rank_tol * singular[0])))
        rank = min(max_bond, numerical_rank)
        cores[site] = vh[:rank].reshape(rank, 2, right_rank)
        discarded[site - 1] = float(np.sum(singular[rank:] ** 2))
        residual = u[:, :rank] * singular[:rank]
        right_rank = rank
    first = residual.reshape(1, 2, right_rank)
    retained_norm = float(np.linalg.norm(first))
    if retained_norm < 1e-15:
        raise ArithmeticError("SVD truncation produced a near-zero state")
    cores[0] = first / retained_norm
    out = tuple(cores)
    state_approx = mps_to_statevector(out)
    fidelity = float(abs(np.vdot(psi, state_approx)) ** 2)
    return MPSDecomposition(
        cores=out, retained_norm=retained_norm,
        discarded_weights=tuple(discarded),
        bond_dimensions=tuple(core.shape[2] for core in out[:-1]),
        fidelity=fidelity,
    )
