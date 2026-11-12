"""Topology representation, canonical forms and mutation operators.

A topology is a tuple of (a, b) pairs with a < b, applied in order to
|0...0>. Two topologies that differ only by swapping adjacent blocks on
disjoint pairs implement exactly the same family of circuits, so the cache
key is the lexicographically smallest topological ordering of the block
dependency DAG.
"""

from __future__ import annotations

import heapq
import json
import random

N = 8
ALL_PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]


def canon_pairs(pairs):
    return tuple((min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs)


def normal_form(top):
    """Lexicographically minimal linear extension of the dependency DAG."""
    top = canon_pairs(top)
    m = len(top)
    preds = [set() for _ in range(m)]
    last = {}
    for i, (a, b) in enumerate(top):
        for q in (a, b):
            if q in last:
                preds[i].add(last[q])
            last[q] = i
    succs = [[] for _ in range(m)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
    indeg = [len(p) for p in preds]
    heap = [(top[i], i) for i in range(m) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        pair, i = heapq.heappop(heap)
        out.append(pair)
        for j in succs[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (top[j], j))
    return tuple(out)


def key(top):
    return json.dumps([list(p) for p in normal_form(top)], separators=(",", ""))


def touches_all(top, n=N):
    return len({q for p in top for q in p}) == n


def redundant_blocks(top):
    """Indices of blocks that can be merged into the next block: the next
    block acting on *both* of its qubits is the same block on the same pair."""
    top = canon_pairs(top)
    out = []
    for i, (a, b) in enumerate(top):
        nxt_a = next((j for j in range(i + 1, len(top)) if a in top[j]), None)
        nxt_b = next((j for j in range(i + 1, len(top)) if b in top[j]), None)
        if nxt_a is not None and nxt_a == nxt_b and top[nxt_a] == (a, b):
            out.append(i)
    return out


def simplify(top):
    """Remove mergeable blocks (each removal never reduces expressivity)."""
    top = list(canon_pairs(top))
    while True:
        r = redundant_blocks(top)
        if not r:
            return tuple(top)
        del top[r[0]]


def depth(top):
    level = {}
    d = 0
    for a, b in canon_pairs(top):
        layer = max(level.get(a, 0), level.get(b, 0)) + 1
        level[a] = level[b] = layer
        d = max(d, layer)
    return d


def valid(top, n=N):
    return len(top) > 0 and touches_all(top, n) and not redundant_blocks(top)


# ----------------------------------------------------------------- mutations

def _rand_pair(rng, weights=None):
    if weights:
        pairs, w = zip(*weights.items())
        return rng.choices(pairs, weights=w)[0]
    return rng.choice(ALL_PAIRS)


def mut_insert(top, rng, weights=None):
    t = list(top)
    t.insert(rng.randrange(len(t) + 1), _rand_pair(rng, weights))
    return t


def mut_delete(top, rng, weights=None):
    t = list(top)
    if len(t) > 1:
        del t[rng.randrange(len(t))]
    return t


def mut_replace(top, rng, weights=None):
    t = list(top)
    i = rng.randrange(len(t))
    if rng.random() < 0.5:   # change one endpoint
        a, b = t[i]
        keep = rng.choice((a, b))
        other = rng.choice([q for q in range(N) if q not in (a, b)])
        t[i] = (min(keep, other), max(keep, other))
    else:
        t[i] = _rand_pair(rng, weights)
    return t


def mut_move(top, rng, weights=None):
    t = list(top)
    g = t.pop(rng.randrange(len(t)))
    t.insert(rng.randrange(len(t) + 1), g)
    return t


def mut_swap_adjacent(top, rng, weights=None):
    t = list(top)
    # only swaps of non-commuting neighbours change the circuit family
    cands = [i for i in range(len(t) - 1) if set(t[i]) & set(t[i + 1]) and t[i] != t[i + 1]]
    if cands:
        i = rng.choice(cands)
        t[i], t[i + 1] = t[i + 1], t[i]
    return t


def mut_relabel_qubit(top, rng, weights=None):
    """Swap the roles of two qubits within a contiguous segment."""
    t = list(top)
    x, y = rng.sample(range(N), 2)
    i = rng.randrange(len(t))
    j = rng.randrange(i, len(t)) + 1
    f = {x: y, y: x}
    for k in range(i, j):
        a, b = (f.get(q, q) for q in t[k])
        t[k] = (min(a, b), max(a, b))
    return t


def crossover_one_point(a, b, rng):
    i = rng.randrange(1, len(a)) if len(a) > 1 else 0
    j = rng.randrange(1, len(b)) if len(b) > 1 else 0
    return list(a[:i]) + list(b[j:])


def crossover_aligned(a, b, rng):
    """Cut both parents at the same relative position (keeps size near)."""
    frac = rng.random()
    return list(a[:round(frac * len(a))]) + list(b[round(frac * len(b)):])


def crossover_segment(a, b, rng):
    """Replace a segment of a with a segment of b of equal length."""
    if len(a) < 2 or len(b) < 2:
        return list(a)
    L = rng.randrange(1, min(len(a), len(b)))
    i = rng.randrange(len(a) - L + 1)
    j = rng.randrange(len(b) - L + 1)
    return list(a[:i]) + list(b[j:j + L]) + list(a[i + L:])


MUTATIONS = {
    "insert": mut_insert, "delete": mut_delete, "replace": mut_replace,
    "move": mut_move, "swap": mut_swap_adjacent, "relabel": mut_relabel_qubit,
}
CROSSOVERS = {
    "one_point": crossover_one_point, "aligned": crossover_aligned,
    "segment": crossover_segment,
}


def random_topology(m, rng, weights=None, n=N):
    while True:
        t = tuple(_rand_pair(rng, weights) for _ in range(m))
        if valid(t, n):
            return t


def repair(top, target_m, rng, weights=None):
    """Simplify, enforce that every qubit is touched and length == target_m."""
    t = list(simplify(top))
    for _ in range(200):
        missing = [q for q in range(N) if not any(q in p for p in t)]
        if missing:
            q = missing[0]
            o = rng.choice([x for x in range(N) if x != q])
            t.insert(rng.randrange(len(t) + 1), (min(q, o), max(q, o)))
            continue
        if target_m is not None and len(t) > target_m:
            del t[rng.randrange(len(t))]
            continue
        if target_m is not None and len(t) < target_m:
            t.insert(rng.randrange(len(t) + 1), _rand_pair(rng, weights))
            continue
        t = list(simplify(t))
        if valid(t) and (target_m is None or len(t) == target_m):
            return tuple(t)
    return None
