"""Generate nearby real, nonnegative quantum states directly in amplitude space.

No Heston parameter perturbation. Each synthetic state lies at a controlled
Fubini--Study distance from a historical state and has unit L2 norm.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AugmentedStates:
    states: np.ndarray                 # (num_new, 2**n), float64
    parent_indices: np.ndarray         # index of original state for each sample
    fs_distances: np.ndarray           # radians, measured from parent states
    max_fs_distance: float             # radius used (radians)


def _validated_states(states: np.ndarray) -> np.ndarray:
    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("states must have shape (number_of_states, 2**num_qubits).")
    if array.shape[1] & (array.shape[1] - 1):
        raise ValueError("State dimension must be a power of two.")
    if not np.isfinite(array).all() or np.min(array) < -1e-12:
        raise ValueError("This augmenter expects finite nonnegative real amplitudes.")
    array = np.maximum(array, 0)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0):
        raise ValueError("At least one state has zero norm.")
    return array / norms[:, None]


def fubini_study_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise, row-aligned Fubini--Study distance for real unit vectors."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Both inputs must have matching (N, D) shapes.")
    overlap = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    return np.arccos(np.clip(np.abs(overlap), 0.0, 1.0))


def historical_radius(states: np.ndarray, quantile: float = 0.75) -> float:
    """Quantile of consecutive historical state distances (chronological input)."""
    states = _validated_states(states)
    if len(states) < 2:
        raise ValueError("At least two historical states are needed to infer a radius.")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")
    steps = fubini_study_distances(states[:-1], states[1:])
    radius = float(np.quantile(steps, quantile))
    if radius < 1e-7:
        raise ValueError("Historical transitions are effectively zero; specify max_fs_distance.")
    return radius


def _random_reference(anchor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A nonnegative *amplitude-space* variation with smooth bin-wise noise."""
    size = anchor.size
    white = rng.normal(size=size)
    half_window = min(8, (size - 1) // 2)
    offsets = np.arange(-half_window, half_window + 1)
    kernel = np.exp(-0.5 * (offsets / 2.5) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(white, kernel, mode="same")
    weights = anchor * anchor
    smooth -= np.sum(weights * smooth)
    variance = np.sum(weights * smooth * smooth)
    if variance < 1e-14:
        smooth = rng.normal(size=size)
        smooth -= np.sum(weights * smooth)
        variance = np.sum(weights * smooth * smooth)
    smooth /= np.sqrt(max(variance, 1e-14))
    # Positive multiplicative amplitude perturbation; no model parameters.
    strength = rng.uniform(1.1, 2.1)
    reference = anchor * np.exp(np.clip(strength * smooth, -25, 25))
    return reference / np.linalg.norm(reference)


def generate_synthetic_states(
    states: np.ndarray,
    num_new_states: int,
    *,
    max_fs_distance: float | None = None,
    radius_quantile: float = 0.75,
    seed: int = 42,
) -> AugmentedStates:
    """Generate exactly num_new_states around the supplied anchor states.

    Samples a radius uniformly from 10%-100% of max_fs_distance.
    By default radius = 75th percentile of *training-only* consecutive
    Fubini--Study distances. Each new state uses spherical interpolation
    towards a random nonnegative amplitude-space reference, which preserves
    nonnegativity and achieves the requested Fubini--Study radius exactly.

    Parent selection is balanced over all anchors and reproducibly shuffled.
    Input states must be chronological for the inferred radius to be meaningful.
    """
    anchors = _validated_states(states)
    if not isinstance(num_new_states, int) or num_new_states < 0:
        raise ValueError("num_new_states must be a nonnegative integer.")
    if max_fs_distance is None:
        radius = historical_radius(anchors, radius_quantile)
    else:
        radius = float(max_fs_distance)
    if not np.isfinite(radius) or not (0 < radius < np.pi / 2):
        raise ValueError("max_fs_distance must be between 0 and pi/2 radians.")

    rng = np.random.default_rng(seed)
    parents = np.resize(np.arange(len(anchors), dtype=np.int64), num_new_states)
    rng.shuffle(parents)
    synthetic = np.empty((num_new_states, anchors.shape[1]), dtype=np.float64)
    achieved = np.empty(num_new_states, dtype=np.float64)

    for j, idx in enumerate(parents):
        anchor = anchors[idx]
        delta = float(rng.uniform(0.10 * radius, radius))
        for trial in range(25):
            reference = _random_reference(anchor, rng)
            gamma = float(np.arccos(np.clip(np.dot(anchor, reference), 0.0, 1.0)))
            if gamma > delta + 1e-8:
                break
        else:
            # Fallback for unusually narrow input states / large requested radii.
            reference = np.full_like(anchor, 1 / np.sqrt(anchor.size))
            gamma = float(np.arccos(np.clip(np.dot(anchor, reference), 0.0, 1.0)))
            if gamma <= delta:
                raise ValueError("Cannot reach requested radius; try a smaller max_fs_distance.")

        # Spherical interpolation: geodesic of length delta from anchor.
        new_state = (
            np.sin(gamma - delta) / np.sin(gamma) * anchor
            + np.sin(delta) / np.sin(gamma) * reference
        )
        new_state = np.maximum(new_state, 0)
        new_state /= np.linalg.norm(new_state)
        synthetic[j] = new_state
        achieved[j] = np.arccos(np.clip(np.dot(anchor, new_state), 0.0, 1.0))

    if num_new_states and not np.allclose(achieved, np.clip(achieved, 0, radius), atol=1e-10):
        raise AssertionError("Generated state fell outside the specified radius.")
    return AugmentedStates(synthetic, parents, achieved, radius)
