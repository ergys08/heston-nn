"""Step-length-matched random walks for real-amplitude quantum states.

The default signed-real control remains unchanged: independent isotropic
Gaussian tangent directions generate a geodesic walk on a fixed real sphere
(S^255 or S^5, respectively).

Set nonnegative=True to keep EVERY amplitude >= 0. This is a different,
positive-orthant geometric control: choose independent nonnegative reference
states and follow the great-circle arc toward each reference by exactly the
historical Fubini--Study step length. The reference-based direction law is
NOT uniform on the tangent sphere conditioned on positivity.

For restricted dimension d<ambient, the positive walk stays within the fixed
d-dimensional linear span of psi0 and d-1 randomly selected coordinate
vectors; its intersection with the positive orthant is not a full S^(d-1).
Consequently positive-orthant and unrestricted S^5/S^255 controls must be
identified separately in figures/tables.
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


def _positive_target(rng, initial_state, allowed_indices):
    """Sample a strictly nonnegative reference in the chosen fixed subspace.

    If allowed_indices is None, draw uniformly on the full positive
    orthant of the sphere (absolute isotropic Gaussian).
    Otherwise draw a nonnegative mixture of psi0 and positive coordinate
    vectors; this retains a fixed linear subspace of the requested dimension.
    """
    if allowed_indices is None:
        vec = np.abs(rng.standard_normal(initial_state.size))
    else:
        coeff = rng.dirichlet(np.full(len(allowed_indices) + 1, 0.35))
        vec = coeff[0] * initial_state.copy()
        vec[allowed_indices] += coeff[1:]
    return vec / np.linalg.norm(vec)


def _positive_geodesic_step(current, angle, rng, initial_state, allowed_indices):
    """Take a step of exact geodesic length angle with nonnegative endpoint.

    The great-circle segment between two nonnegative endpoints remains in the
    nonnegative orthant. We generate a random positive reference far enough
    away, then move along that segment. Rejection is on the REFERENCE, not on
    the tangent Gaussian, and does not imply a uniform conditional direction.

    A coordinate-vertex fallback ensures feasibility for modest step sizes;
    raise an explicit exception if the chosen fixed subspace cannot supply
    the requested step. Never silently shorten a historical displacement.
    """
    cos_angle = float(np.cos(angle))
    for _ in range(64):
        target = _positive_target(rng, initial_state, allowed_indices)
        overlap = float(np.clip(np.dot(current, target), -1.0, 1.0))
        if overlap <= cos_angle - 1e-12:
            break
    else:
        # Every e_j is a valid nonnegative reference in the chosen subspace.
        candidates = np.arange(current.size) if allowed_indices is None else allowed_indices
        feasible = candidates[current[candidates] <= cos_angle - 1e-12]
        if not len(feasible):
            raise ValueError(
                "Cannot realise this step in the chosen nonnegative subspace. "
                "Increase subspace_dimension or change the random seed."
            )
        target = np.zeros_like(current)
        target[int(rng.choice(feasible))] = 1.0
        overlap = float(np.dot(current, target))

    separation = float(np.arccos(np.clip(overlap, -1.0, 1.0)))
    # Spherical linear interpolation, with nonnegative weights.
    left = np.sin(separation - angle) / np.sin(separation)
    right = np.sin(angle) / np.sin(separation)
    endpoint = left * current + right * target
    # Only remove errors at machine precision; don't distort genuine steps.
    if endpoint.min() < -1e-12:
        raise ArithmeticError("Nonnegative geodesic step failed")
    endpoint = np.maximum(endpoint, 0.0)
    endpoint /= np.linalg.norm(endpoint)
    return endpoint


def generate_matched_random_walk(
    initial_state: np.ndarray,
    step_lengths: np.ndarray,
    *,
    seed: int = 12345,
    subspace_dimension: int | None = None,
    nonnegative: bool = False,
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
    if nonnegative and np.any(psi0 < -1e-12):
        raise ValueError("nonnegative=True requires a nonnegative initial state")
    if nonnegative:
        psi0 = np.maximum(psi0, 0.0)
        psi0 /= np.linalg.norm(psi0)
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

    if nonnegative:
        # A positive orthant has no nontrivial translation-invariant direction
        # distribution. The restricted walk uses a fixed positive cone whose
        # linear span has the requested dimension (including psi0).
        allowed_indices = (
            None if dimension == ambient else
            rng.choice(ambient, size=dimension - 1, replace=False)
        )
        current = psi0.copy()
        for index, angle in enumerate(delta, start=1):
            if angle > 0:
                current = _positive_geodesic_step(
                    current, float(angle), rng, psi0, allowed_indices
                )
            result[index] = current
        return result

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
    nonnegative: bool = False,
) -> np.ndarray:
    """Match daily jumps of a historical real-amplitude state trajectory."""
    states = _normalized_real_states(historical_states)
    return generate_matched_random_walk(
        states[0],
        consecutive_step_lengths(states),
        seed=seed,
        subspace_dimension=subspace_dimension,
        nonnegative=nonnegative,
    )
