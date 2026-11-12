"""Rigorous upper bounds on achievable fidelity for <= 7 two-qubit blocks.

Any topology with M <= 7 blocks that touches all 8 qubits has an
interaction graph that is either (i) a spanning tree (exactly 7 distinct
edges, connected) or (ii) disconnected. (A topology leaving a qubit
untouched gives that qubit |0>, fidelity <= P(bit=0) for that qubit.)

(i) Removing tree edge e splits the qubits into (A, B); only the block on e
    acts across the cut, so the prepared state has Schmidt rank <= 2 there
    and F <= s2(A|B) = lambda_1^2 + lambda_2^2. Hence, for a tree,
    F <= min_e s2(cut_e), and over all trees F <= max_T min_e s2(cut_e).
(ii) The state is a product across some bipartition with both sides >= 2
    qubits... or, more generally, any nontrivial bipartition along the
    component structure, so F <= max_{A} lambda_1^2(A|B).
Untouched qubit: F <= max_q P(bit_q = 0).

Both bounds hold for arbitrary (complex) unitaries and any gate order.
"""
import itertools, json
from pathlib import Path
import numpy as np

N = 8


def schmidt_sq(psi, A):
    t = psi.reshape([2] * N)          # axis k <-> qubit N-1-k
    axes_a = [N - 1 - q for q in A]
    axes_b = [k for k in range(N) if k not in axes_a]
    m = np.transpose(t, axes_a + axes_b).reshape(1 << len(A), -1)
    return np.sort(np.linalg.svd(m, compute_uv=False) ** 2)[::-1]


def prufer_trees(n=N):
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for x in seq:
            degree[x] += 1
        edges = []
        seq = list(seq)
        for x in seq:
            leaf = next(i for i in range(n) if degree[i] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, v = [i for i in range(n) if degree[i] == 1]
        edges.append((u, v))
        yield edges


def cut_side(edges, e):
    """Bitmask of the component containing e[0] after removing edge e."""
    adj = {i: [] for i in range(N)}
    for a, b in edges:
        if (a, b) != e:
            adj[a].append(b); adj[b].append(a)
    seen, stack = {e[0]}, [e[0]]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    mask = sum(1 << q for q in seen)
    return min(mask, (1 << N) - 1 - mask)   # canonical side


def main():
    amps = np.load(Path(__file__).resolve().parent.parent / "aqce_results_2013" / "target_states.npz")["amplitudes"]
    S = len(amps)
    masks = [m for m in range(1, 1 << N) if m < (1 << N) - 1 - m]   # 127 bipartitions
    s1 = np.zeros((S, 1 << N)); s2 = np.zeros((S, 1 << N))
    for m in masks:
        A = [q for q in range(N) if m >> q & 1]
        for i in range(S):
            sv = schmidt_sq(amps[i], A)
            s1[i, m] = sv[0]; s2[i, m] = sv[:2].sum()
    # (i) trees: enumerate all 8^6 labelled spanning trees
    tree_cuts = np.array([[cut_side(t, e) for e in t] for t in prufer_trees()])  # (262144, 7)
    tree_bound = np.zeros(S); best_tree = [None] * S
    uniq, inv = np.unique(np.sort(tree_cuts, axis=1), axis=0, return_inverse=True)
    for i in range(S):
        vals = s2[i][uniq].min(axis=1)
        j = int(np.argmax(vals)); tree_bound[i] = vals[j]
    # (ii) disconnected: product across bipartition with both sides >= 2 qubits
    big = [m for m in masks if 2 <= bin(m).count("1") <= N - 2]
    prod_bound = s1[:, big].max(axis=1)
    # untouched qubit
    p0 = np.array([[np.sum(np.abs(a.reshape([2]*N)).take(0, axis=N-1-q) ** 2) for q in range(N)] for a in amps]).max(1)
    bound7 = np.maximum.reduce([tree_bound, prod_bound, p0])
    bound6 = np.maximum(prod_bound, p0)
    out = dict(
        description=__doc__,
        n_states_where_M7_cannot_reach_0995=int(np.sum(bound7 < 0.995)),
        states_where_M7_cannot_reach_0995=np.where(bound7 < 0.995)[0].tolist(),
        bound7_values_for_those={int(i): float(bound7[i]) for i in np.where(bound7 < 0.995)[0]},
        n_states_where_M6_cannot_reach_0995=int(np.sum(bound6 < 0.995)),
        min_tree_bound=float(tree_bound.min()), min_product_bound=float(prod_bound.max(initial=0) and prod_bound.min()),
        tree_bound=tree_bound.tolist(), product_bound=prod_bound.tolist(), untouched_bound=p0.tolist(),
    )
    path = Path(__file__).resolve().parent / "results" / "lower_bound_M7.json"
    path.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("tree_bound", "product_bound", "untouched_bound", "description")})
    order = np.argsort(bound7)[:8]
    for i in order:
        print(i, "tree %.6f product %.6f untouched %.4f" % (tree_bound[i], prod_bound[i], p0[i]))


if __name__ == "__main__":
    main()
