"""Schmidt diagnostics using the repository's q0-first MPS convention."""
import numpy as np
from MPS.mps_decomposition import decompose_state, statevector_to_tensor


def spectra(state, all_cuts=True):
    """Squared Schmidt coefficients, keyed by canonical bipartition bitmask."""
    tensor = statevector_to_tensor(state)
    n = tensor.ndim
    full = (1 << n) - 1
    masks = range(1, full) if all_cuts else [(1 << k) - 1 for k in range(1, n)]
    result = {}
    for mask in masks:
        key = min(mask, full ^ mask)
        if key in result:
            continue
        left = [i for i in range(n) if key & (1 << i)]
        right = [i for i in range(n) if not key & (1 << i)]
        mat = tensor.transpose(left + right).reshape(1 << len(left), -1)
        weights = np.linalg.svd(mat, compute_uv=False) ** 2
        result[key] = weights / weights.sum()
    return result


def effective_rank(weights, eps):
    """Smallest rank whose discarded squared Schmidt weight is <= eps."""
    tails = np.r_[np.cumsum(weights[::-1])[::-1], 0.]
    return int(np.flatnonzero(tails[1:] <= eps)[0] + 1)


def entropy(weights):
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log2(positive)))


def best_order(weights_by_cut, n, eps):
    """Exact minimax effective Schmidt rank over all linear qubit orders.

    Subset DP: append a qubit, paying the rank of the resulting cut.
    Lexicographic tie breaking makes the natural order preferred if optimal.
    This optimises a cut lower bound, not global TT-SVD fidelity.
    """
    full = (1 << n) - 1
    ranks = {m: effective_rank(w, eps) for m, w in weights_by_cut.items()}
    dp = {0: (1, ())}
    for mask in range(1, full + 1):
        cost = 1 if mask == full else ranks[min(mask, full ^ mask)]
        candidates = []
        for q in range(n):
            bit = 1 << q
            if mask & bit:
                previous, order = dp[mask ^ bit]
                candidates.append((max(cost, previous), order + (q,)))
        dp[mask] = min(candidates)
    # DP finds the exact minimax cost; reconstruct the lexicographically first
    # order satisfying that global threshold (local tie breaking alone does
    # not guarantee the globally lexicographically first optimal order).
    optimum = dp[full][0]
    can_finish = {full: True}
    for mask in range(full - 1, -1, -1):
        can_finish[mask] = any(
            can_finish[mask | (1 << q)] and (
                (mask | (1 << q)) == full or
                ranks[min(mask | (1 << q), full ^ (mask | (1 << q)))] <= optimum
            )
            for q in range(n) if not mask & (1 << q)
        )
    order = []
    mask = 0
    while mask != full:
        for q in range(n):
            if mask & (1 << q):
                continue
            nxt = mask | (1 << q)
            if can_finish[nxt] and (nxt == full or ranks[min(nxt, full ^ nxt)] <= optimum):
                order.append(q)
                mask = nxt
                break
    return optimum, tuple(order)


def reorder_state(state, order):
    """Relabel physical axes so TT-SVD follows the given original qubit order."""
    tensor = statevector_to_tensor(state).transpose(order)
    return tensor.transpose(tuple(reversed(range(tensor.ndim)))).reshape(-1)


def fidelity_curve(state):
    """Actual global squared overlaps, for every integer bond cap."""
    n = int(np.asarray(state).size).bit_length() - 1
    return np.array([
        np.clip(decompose_state(state, max_bond=r, rank_tol=0).fidelity, 0, 1)
        for r in range(1, (1 << (n // 2)) + 1)
    ])


def first_passing(curve, eps):
    hits = np.flatnonzero(1 - curve <= eps)
    if not len(hits):
        raise ArithmeticError("Even maximum-rank TT-SVD failed target fidelity")
    return int(hits[0] + 1)


def analyse(state, tolerances, all_cuts=True):
    n = int(np.asarray(state).size).bit_length() - 1
    full = (1 << n) - 1
    natural = tuple(range(n))
    weights = spectra(state, all_cuts)
    cuts = [weights[(1 << k) - 1] for k in range(1, n)]
    curve = fidelity_curve(state)
    curves = {natural: curve}
    rows = []
    for eps in tolerances:
        profile = [effective_rank(w, eps) for w in cuts]
        lower = max(profile)
        cap = first_passing(curve, eps)
        if cap < lower:
            raise ArithmeticError("TT-SVD fidelity inconsistent with cut bound")
        row = dict(epsilon=eps, natural_cut_bound=lower, natural_profile=profile,
                   natural_tt_cap=cap, natural_tt_fidelity=float(curve[cap-1]))
        if all_cuts:
            optimal, order = best_order(weights, n, eps)
            if order not in curves:
                curves[order] = fidelity_curve(reorder_state(state, order))
            candidate_curve = curves[order]
            candidate = first_passing(candidate_curve, eps)
            if candidate < optimal:
                raise ArithmeticError("Reordered TT-SVD inconsistent with cut bound")
            row.update(best_cut_bound=optimal, best_cut_order=list(order),
                       reordered_tt_cap=candidate,
                       reordered_tt_fidelity=float(candidate_curve[candidate-1]),
                       best_tested_tt_cap=min(cap, candidate),
                       cut_order_gap=lower-optimal)
        rows.append(row)
    return weights, curves, rows
