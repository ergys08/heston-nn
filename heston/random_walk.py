"""Step-length-matched random walks through real quantum-state space.

A path of normalized real amplitude vectors follows geodesics on the unit
sphere. For every historical step of Fubini--Study length delta_t, choose
an independent random tangent direction v_t and update via

    psi[t+1] = cos(delta_t) * psi[t] + sin(delta_t) * v_t.

The walk starts at the actual first Heston state, preserves the ordered
list of historical Fubini--Study step lengths, and has the same number of
states. The resulting signed-amplitude states are valid pure quantum states,
but are generally NOT amplitude encodings of Heston PDFs.

Use subspace_dimension=6 to compare a random walk on a fixed five-
dimensional sphere with one on the full 256-dimensional real unit sphere
(subspace_dimension=256 for eight qubits). The sphere is a geometric
control, NOT the Heston model manifold.
"""

from __future__ import annotations

import numpy as np


def _normalized_real_states(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError("states must have shape (n_states, dimension >= 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("states contain NaN or infinity")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("all states must be nonzero")
    return x / norms[:, None]


def consecutive_step_lengths(states: np.ndarray) -> np.ndarray:
    """Historical Fubini--Study distances, in chronological order (radians)."""
    x = _normalized_real_states(states)
    overlaps = np.einsum("ij,ij->i", x[:-1], x[1:])
    return np.arccos(np.clip(np.abs(overlaps), 0.0, 1.0))


def generate_matched_random_walk(
    initial_state: np.ndarray,
    step_lengths: np.ndarray,
    *,
    seed: int = 12345,
    subspace_dimension: int | None = None,
) -> np.ndarray:
    """Generate one matched walk, returning (n_steps + 1, n_amplitudes).

    subspace_dimension is the dimension of the linear real subspace
    containing the walk. Its unit sphere has intrinsic dimension one less.
    A fixed random subspace is constructed through the initial state.
    None means the full ambient real-amplitude space.
    """
    psi0 = np.asarray(initial_state, dtype=np.float64)
    if psi0.ndim != 1 or psi0.size < 2 or not np.all(np.isfinite(psi0)):
        raise ValueError("initial_state must be a finite real vector of length >= 2")
    nrm = np.linalg.norm(psi0)
    if nrm == 0:
        raise ValueError("initial_state cannot be zero")
    psi0 = psi0 / nrm
    delta = np.asarray(step_lengths, dtype=np.float64)
    if delta.ndim != 1 or not np.all(np.isfinite(delta)):
        raise ValueError("step_lengths must be a finite one-dimensional array")
    if np.any(delta < 0) or np.any(delta > np.pi / 2 + 1e-12):
        raise ValueError("Fubini--Study step lengths must lie in [0, pi/2]")

    ambient = psi0.size
    dimension = ambient if subspace_dimension is None else subspace_dimension
    if not isinstance(dimension, (int, np.integer)) or not 2 <= dimension <= ambient:
        raise ValueError("subspace_dimension must be between 2 and the state dimension")

    rng = np.random.default_rng(seed)
    result = np.empty((delta.size + 1, ambient), dtype=np.float64)
    result[0] = psi0

    if dimension < ambient:
        # Orthonormal basis of a fixed random subspace containing psi0.
        raw = rng.standard_normal((ambient, dimension - 1))
        raw -= psi0[:, None] * (psi0 @ raw)[None, :]
        basis_tangent, _ = np.linalg.qr(raw, mode="reduced")
        basis = np.column_stack((psi0, basis_tangent))
        current = np.zeros(dimension, dtype=np.float64)
        current[0] = 1.0
    else:
        basis = None
        current = psi0.copy()

    for index, angle in enumerate(delta, start=1):
        if angle == 0.0:
            result[index] = result[index - 1]
            continue
        direction = rng.standard_normal(dimension)
        direction -= current * np.dot(direction, current)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ArithmeticError("Failed to sample a nonzero tangent vector")
        direction /= norm
        current = np.cos(angle) * current + np.sin(angle) * direction
        current /= np.linalg.norm(current)
        result[index] = basis @ current if basis is not None else current

    return result


def matched_step_random_walk(
    historical_states: np.ndarray,
    *,
    seed: int = 12345,
    subspace_dimension: int | None = None,
) -> np.ndarray:
    """Match daily jumps of a historical real-amplitude state trajectory."""
    states = _normalized_real_states(historical_states)
    return generate_matched_random_walk(
        states[0],
        consecutive_step_lengths(states),
        seed=seed,
        subspace_dimension=subspace_dimension,
    )
