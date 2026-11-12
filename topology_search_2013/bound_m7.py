"""Rigorous fidelity upper bound for every 7-block topology (wire-cut bound).

Disconnected 7-block topologies are excluded by lower_bound.py (product
bound < 0.995 for all states). A connected 7-block topology on 8 qubits has
a spanning-tree interaction graph; its circuit tensor network then has 7
gate nodes joined by 2*7 - 8 = 6 internal qubit wires and is itself a tree.
Cutting a wire splits the network into two subtrees, so the output state
has Schmidt rank <= 2 across the induced bipartition of physical qubits,
giving F <= lambda_1^2 + lambda_2^2 =: s2(cut) for each of the 6 cuts.

We enumerate a SUPERSET of all such networks: every labelled tree on 7
nodes with node degree d <= 4, every number of physical outputs per node
l = 2 - out (out = outgoing wires, 0 <= out <= 2, in = d - out <= 2),
sum l = 8, and every assignment of the 8 qubits to the physical slots.
For each state, bound = max over networks of min over its 6 cuts of s2.
If bound < 0.995 for any state, no 7-block topology reaches 250/250.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from .lower_bound import schmidt_sq

N = 8


def prufer_to_edges(seq, n):
    degree = [1] * n
    for x in seq:
        degree[x] += 1
    edges = []
    for x in seq:
        leaf = next(i for i in range(n) if degree[i] == 1)
        edges.append((leaf, x))
        degree[leaf] -= 1
        degree[x] -= 1
    u, v = [i for i in range(n) if degree[i] == 1]
    return edges + [(u, v)]


def structures():
    """Distinct cut structures: tuples of 6 frozensets of slot indices."""
    seen = set()
    for seq in itertools.product(range(7), repeat=5):
        edges = prufer_to_edges(seq, 7)
        deg = [0] * 7
        for a, b in edges:
            deg[a] += 1; deg[b] += 1
        if max(deg) > 4:
            continue
        options = [[2 - out for out in range(max(0, d - 2), min(2, d) + 1)] for d in deg]
        for leaves in itertools.product(*options):
            if sum(leaves) != N:
                continue
            slots, k = [], 0
            for l in leaves:
                slots.append(list(range(k, k + l))); k += l
            cuts = []
            for e in edges:
                adj = {i: [] for i in range(7)}
                for a, b in edges:
                    if (a, b) != e:
                        adj[a].append(b); adj[b].append(a)
                side, stack = {e[0]}, [e[0]]
                while stack:
                    x = stack.pop()
                    for y in adj[x]:
                        if y not in side:
                            side.add(y); stack.append(y)
                s = frozenset(q for i in side for q in slots[i])
                if s.__len__() > N // 2 or (len(s) == N // 2 and 0 not in s):
                    s = frozenset(range(N)) - s
                cuts.append(s)
            nontriv = tuple(sorted(tuple(sorted(c)) for c in cuts if 1 < len(c) < N - 1 or len(c) == 0))
            seen.add(nontriv)
    return seen


def canonical_mask(qubits):
    m = sum(1 << q for q in qubits)
    full = (1 << N) - 1
    return min(m, full - m)


def main():
    root = Path(__file__).resolve().parent
    amps = np.load(root.parent / "aqce_results_2013" / "target_states.npz")["amplitudes"]
    structs = [s for s in structures()]
    print("distinct cut structures:", len(structs))
    perms = np.array(list(itertools.permutations(range(N))))          # (40320, 8)
    # mask lookup for every structure under every slot->qubit permutation
    s2 = np.ones((len(amps), 1 << N))
    for m in range(1, (1 << N) - 1):
        A = [q for q in range(N) if m >> q & 1]
        if 2 <= len(A) <= N - 2:
            for i in range(len(amps)):
                s2[i, m] = schmidt_sq(amps[i], A)[:2].sum()
    struct_masks = []
    pw = (1 << perms)                                                  # (P, 8): qubit bit for slot
    full = (1 << N) - 1
    for s in structs:
        cols = []
        for cut in s:
            m = pw[:, list(cut)].sum(1) if cut else np.zeros(len(perms), int)
            cols.append(np.minimum(m, full - m))
        struct_masks.append(np.stack(cols, 1) if cols else np.zeros((len(perms), 0), int))
    bound = np.zeros(len(amps))
    for i in range(len(amps)):
        best = 0.0
        for sm in struct_masks:
            if sm.shape[1] == 0:
                best = 1.0; break
            best = max(best, s2[i][sm].min(1).max())
        bound[i] = best
    fail = np.where(bound < 0.995)[0]
    out = dict(description=__doc__, n_structures=len(structs),
               states_where_M7_cannot_reach_0995=fail.tolist(),
               bounds_of_those={int(i): float(bound[i]) for i in fail},
               bound=bound.tolist())
    (root / "results" / "bound_M7_wirecut.json").write_text(json.dumps(out, indent=1))
    print("states provably unreachable with 7 blocks:", fail.tolist(), [round(bound[i], 6) for i in fail])
    print("lowest bounds:", [(int(i), round(bound[i], 6)) for i in np.argsort(bound)[:8]])


if __name__ == "__main__":
    main()
