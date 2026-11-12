"""Naive quantum-state control: 1,000 random amplitude vectors near the first state.

Directions are independent and unsmoothed. The states are real, nonnegative,
normalised, and their Fubini--Study distance from the anchor is controlled.
They are NOT sampled from Heston parameter space or Haar-uniformly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RandomStateSample:
    states: np.ndarray
    fs_distances: np.ndarray
    max_fs_distance: float
    seed: int
    min_radius_fraction: float


def generate_random_states(
    first_state: np.ndarray,
    num_states: int = 1000,
    *,
    max_fs_distance: float = 0.10095073648716599,
    min_radius_fraction: float = 0.10,
    seed: int = 42,
) -> RandomStateSample:
    """Sample states by moving from first_state toward iid half-normal references.

    For each sample, draw r_j=abs(N(0,1)), normalise r, then move a distance
    delta ~ U(min_radius_fraction*R,R) from the anchor toward r along a
    spherical geodesic. This preserves positivity and has exact FS radius.
    The default R matches the historical-local baseline for comparison.
    """
    psi = np.asarray(first_state, dtype=np.float64).copy()
    if psi.ndim != 1 or psi.size < 2 or psi.size & (psi.size - 1):
        raise ValueError("first_state must be a (2**num_qubits,) vector")
    if not np.all(np.isfinite(psi)) or np.min(psi) < -1e-12:
        raise ValueError("first_state must contain finite nonnegative real amplitudes")
    psi = np.maximum(psi, 0.0)
    norm = np.linalg.norm(psi)
    if norm <= 0:
        raise ValueError("first_state has zero norm")
    psi /= norm
    if not isinstance(num_states, int) or num_states < 1:
        raise ValueError("num_states must be a positive integer")
    if not (np.isfinite(max_fs_distance) and 0 < max_fs_distance < np.pi / 2):
        raise ValueError("max_fs_distance must lie between 0 and pi/2")
    if not (0 <= min_radius_fraction <= 1):
        raise ValueError("min_radius_fraction must be in [0, 1]")

    rng = np.random.default_rng(seed)
    output = np.empty((num_states, psi.size), dtype=np.float64)
    radii = np.empty(num_states, dtype=np.float64)
    for k in range(num_states):
        ref = np.abs(rng.standard_normal(psi.size))
        ref /= np.linalg.norm(ref)
        phi = np.arccos(np.clip(float(psi @ ref), 0.0, 1.0))
        delta = float(rng.uniform(min_radius_fraction * max_fs_distance, max_fs_distance))
        if phi <= delta:
            raise ValueError("Random reference too close to anchor for requested FS radius")
        state = np.sin(phi-delta)/np.sin(phi)*psi + np.sin(delta)/np.sin(phi)*ref
        state /= np.linalg.norm(state)
        output[k] = state
        radii[k] = np.arccos(np.clip(float(psi @ state), 0.0, 1.0))
    if not np.all(np.isfinite(output)) or np.min(output) < -1e-12:
        raise AssertionError("Invalid random states")
    if not np.allclose(np.linalg.norm(output,axis=1), 1, atol=1e-12):
        raise AssertionError("Random states are not normalised")
    if np.any(radii > max_fs_distance+1e-9):
        raise AssertionError("Random state lies outside chosen radius")
    return RandomStateSample(output,radii,max_fs_distance,seed,min_radius_fraction)


generate_synthetic_random_states = generate_random_states
