"""Fubini--Study covering of finite sets of pure quantum states.

This module computes the sample-centred covering problem. Candidate ball
centres are the supplied states (or explicit candidate_centres), so this is
not necessarily the unrestricted metric covering number, where centres may
be arbitrary points in projective Hilbert space.

method='exact' solves the finite set-cover integer program with SciPy's
HiGHS MILP solver and only returns a count if optimality is proved.
method='greedy' is a fast valid cover but only an UPPER BOUND on that
sample-centred minimum. Neither count should be silently described as the
unrestricted quantum-state covering number.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoverResult:
    count: int
    centre_indices: tuple[int, ...]
    min_fidelity: float
    method: str
    proven_optimal: bool


def _normalise(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states)
    x = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("states must have shape (n_states >= 1, dimension >= 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("states contain NaN or infinity")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("quantum states must be nonzero")
    return x / norms[:, None]


def fubini_study_distances(states: np.ndarray, centres: np.ndarray | None = None) -> np.ndarray:
    """Return shape (n_states, n_centres) of projective-state distances in radians."""
    targets = _normalise(states)
    refs = targets if centres is None else _normalise(centres)
    if targets.shape[1] != refs.shape[1]:
        raise ValueError("targets and centres have different Hilbert dimensions")
    overlaps = targets.conj() @ refs.T
    return np.arccos(np.clip(np.abs(overlaps), 0.0, 1.0))


def fidelity_matrix(states: np.ndarray, centres: np.ndarray | None = None) -> np.ndarray:
    """Squared quantum-state fidelities for all target/candidate-centre pairs."""
    targets = _normalise(states)
    refs = targets if centres is None else _normalise(centres)
    if targets.shape[1] != refs.shape[1]:
        raise ValueError("targets and centres have different Hilbert dimensions")
    overlaps = targets.conj() @ refs.T
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def covering_from_fidelities(
    fidelities: np.ndarray,
    *,
    min_fidelity: float = 0.999,
    method: str = "greedy",
    time_limit: float | None = None,
) -> CoverResult:
    """Solve a sample-centred cover from a PRECOMPUTED fidelity matrix.

    Input shape (n_targets, n_candidate_centres). Reuse this function when
    evaluating many thresholds on the same trajectory.
    """
    if not np.isfinite(min_fidelity) or not 0.0 < min_fidelity <= 1.0:
        raise ValueError("min_fidelity must lie in (0, 1]")
    fidelities = np.asarray(fidelities)
    if (fidelities.ndim != 2 or 0 in fidelities.shape
            or not np.all(np.isfinite(fidelities))
            or np.any((fidelities < -1e-12) | (fidelities > 1 + 1e-12))):
        raise ValueError("fidelities must be a finite matrix with entries in [0, 1]")
    # Slack only covers floating-point roundoff on nominally identical vectors.
    incidence = fidelities >= min_fidelity - 1e-12
    if np.any(~np.any(incidence, axis=1)):
        raise ValueError("Some target states are not covered by any candidate centre")
    n_targets, n_centres = incidence.shape

    if method == "greedy":
        uncovered = np.ones(n_targets, dtype=bool)
        chosen: list[int] = []
        while np.any(uncovered):
            gains = incidence[uncovered].sum(axis=0)
            centre = int(np.argmax(gains))  # deterministic tie-break
            if gains[centre] == 0:
                raise RuntimeError("Incidence matrix does not cover targets")
            chosen.append(centre)
            uncovered &= ~incidence[:, centre]
        indices = tuple(chosen)
        return CoverResult(len(indices), indices, float(min_fidelity), method, False)

    if method == "exact":
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csc_matrix

        if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive if provided")
        constraints = LinearConstraint(
            csc_matrix(incidence.astype(np.float64)),
            lb=np.ones(n_targets),
            ub=np.full(n_targets, np.inf),
        )
        options = {} if time_limit is None else {"time_limit": float(time_limit)}
        solution = milp(
            c=np.ones(n_centres),
            integrality=np.ones(n_centres),
            bounds=Bounds(np.zeros(n_centres), np.ones(n_centres)),
            constraints=constraints,
            options=options,
        )
        if solution.status != 0 or solution.x is None:
            raise RuntimeError(
                f"MILP did not prove a minimum cover (status {solution.status}): "
                f"{solution.message}. Try 'greedy' or increase time_limit."
            )
        indices = tuple(int(i) for i in np.flatnonzero(solution.x > 0.5))
        if not np.all(np.any(incidence[:, indices], axis=1)):
            raise ArithmeticError("MILP returned a non-covering solution")
        return CoverResult(len(indices), indices, float(min_fidelity), method, True)

    raise ValueError("method must be 'greedy' or 'exact'")


def covering_number(
    states: np.ndarray,
    *,
    min_fidelity: float = 0.999,
    method: str = "greedy",
    candidate_centres: np.ndarray | None = None,
    time_limit: float | None = None,
) -> CoverResult:
    """Find a sample-centred cover at squared fidelity >= min_fidelity.

    With method='greedy': repeatedly select the centre covering the most
    as-yet uncovered states. This is NOT necessarily the minimal cover.

    With method='exact': solve min sum(z_j), subject to each target
    covered by >= 1 chosen centre. Fail explicitly if the MILP cannot prove
    optimality within time_limit (seconds). Requires scipy >= 1.10.

    The returned centre_indices refer to the original candidate-centre order.
    """
    return covering_from_fidelities(
        fidelity_matrix(states, candidate_centres),
        min_fidelity=min_fidelity,
        method=method,
        time_limit=time_limit,
    )
