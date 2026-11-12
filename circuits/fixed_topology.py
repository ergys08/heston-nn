"""Fixed-topology state preparation with arbitrary two-qubit unitary blocks.

A topology is an ordered list of qubit pairs ``(a, b)``; block ``m`` is a
4x4 unitary acting on that pair, and blocks are applied to ``|0...0>`` in
list order. The interaction sequence is shared by every target state, while
the unitaries are optimised independently for each state.

Conventions (identical to qulacs and the saved AQCE results):

* qubit ``q`` is bit ``q`` of the computational-basis index (little endian);
* a block on pair ``(a, b)`` uses the 4x4 matrix index ``2*bit(a) + bit(b)``.
  This equals qulacs ``DenseMatrix([b, a], U)``; AQCE files store the
  qulacs target list ``[b, a]``, so use ``pairs_from_qulacs`` to convert.

The optimiser is the SVD (polar decomposition) environment update used by
AQCE, but the qubit pair of every block stays fixed. Each block update
maximises |<target|C|0>| exactly with all other blocks fixed, so fidelity
never decreases during a sweep. Many states and restarts are optimised
simultaneously as one batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

N_QUBITS = 8


def canonical_topology(pairs):
    """Return pairs as tuples (a, b) with a < b (block action is symmetric
    after relabelling the unitary)."""
    out = []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a == b:
            raise ValueError("A two-qubit block needs two distinct qubits")
        out.append((min(a, b), max(a, b)))
    return out


def pairs_from_qulacs(target_lists):
    """Convert saved qulacs target lists [t0, t1] to (a, b) = (t1, t0)."""
    return [(int(t[1]), int(t[0])) for t in target_lists]


def swap_matrix_qubits(u):
    """Relabel a 4x4 block so it acts on (b, a) instead of (a, b)."""
    p = np.array([0, 2, 1, 3])
    return u[..., p, :][..., :, p]


@lru_cache(maxsize=None)
def pair_permutation(a, b, n=N_QUBITS):
    """Index array that gathers a state into shape (4, 2**(n-2)), with the
    row equal to 2*bit(a) + bit(b)."""
    if not (0 <= a < n and 0 <= b < n) or a == b:
        raise ValueError("Invalid qubit pair")
    rest = [q for q in range(n) if q not in (a, b)]
    idx = np.empty((4, 1 << (n - 2)), dtype=np.int64)
    for r in range(4):
        base = ((r >> 1) & 1) << a | (r & 1) << b
        for c in range(1 << (n - 2)):
            k = base
            for t, q in enumerate(rest):
                k |= ((c >> t) & 1) << q
            idx[r, c] = k
    return idx.reshape(-1)


def apply_block(states, unitaries, pair, n=N_QUBITS, adjoint=False):
    """Apply batched 4x4 unitaries (B,4,4) on ``pair`` to states (B, 2**n)."""
    perm = pair_permutation(*pair, n)
    x = states[:, perm].reshape(states.shape[0], 4, -1)
    u = np.conj(np.swapaxes(unitaries, -1, -2)) if adjoint else unitaries
    out = np.empty_like(states)
    out[:, perm] = (u @ x).reshape(states.shape[0], -1)
    return out


def prepare(topology, unitaries, n=N_QUBITS):
    """Return C|0> for unitaries of shape (B, M, 4, 4)."""
    unitaries = np.asarray(unitaries)
    batch = unitaries.shape[0]
    psi = np.zeros((batch, 1 << n), dtype=unitaries.dtype)
    psi[:, 0] = 1
    for m, pair in enumerate(topology):
        psi = apply_block(psi, unitaries[:, m], pair, n)
    return psi


def fidelities(targets, topology, unitaries, n=N_QUBITS):
    psi = prepare(topology, unitaries, n)
    return np.abs(np.einsum("bk,bk->b", np.conj(targets), psi)) ** 2


def _environment(phi, psi, pair, n):
    """E with <psi| U |phi> = Tr(U E) for a block on ``pair``."""
    perm = pair_permutation(*pair, n)
    f = phi[:, perm].reshape(phi.shape[0], 4, -1)
    g = psi[:, perm].reshape(psi.shape[0], 4, -1)
    return f @ np.conj(np.swapaxes(g, -1, -2))


def _polar_update(env):
    """Unitary maximising |Tr(U E)|, and the resulting overlap sum(S)."""
    w, s, vh = np.linalg.svd(env)
    u = np.conj(np.swapaxes(vh, -1, -2)) @ np.conj(np.swapaxes(w, -1, -2))
    return u, s.sum(-1)


def _reduced_density(psi, pair, n):
    perm = pair_permutation(*pair, n)
    g = psi[:, perm].reshape(psi.shape[0], 4, -1)
    return g @ np.conj(np.swapaxes(g, -1, -2))


# ----------------------------------------------------------------- initialisers

def haar_unitaries(rng, shape, dtype=complex):
    """Haar-random (complex) or orthogonal (real) 4x4 matrices."""
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        z = (rng.standard_normal((*shape, 4, 4))
             + 1j * rng.standard_normal((*shape, 4, 4))) / np.sqrt(2)
    else:
        z = rng.standard_normal((*shape, 4, 4))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / np.abs(d))[..., None, :]


def near_identity_unitaries(rng, shape, scale=0.1, dtype=complex):
    """exp(i*scale*H) with random Hermitian H (or exp(scale*A), A antisymmetric)."""
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        h = rng.standard_normal((*shape, 4, 4)) + 1j * rng.standard_normal((*shape, 4, 4))
        h = (h + np.conj(np.swapaxes(h, -1, -2))) / 2
        w, v = np.linalg.eigh(h)
        return (v * np.exp(1j * scale * w)[..., None, :]) @ np.conj(np.swapaxes(v, -1, -2))
    a = rng.standard_normal((*shape, 4, 4))
    a = (a - np.swapaxes(a, -1, -2)) / 2
    from scipy.linalg import expm
    flat = a.reshape(-1, 4, 4)
    return np.stack([expm(scale * x) for x in flat]).reshape(*shape, 4, 4)


def greedy_disentangle(targets, topology, n=N_QUBITS):
    """AQCE-style initialisation restricted to a fixed topology.

    Walk from the last block to the first; each block is set to the
    eigenbasis of the current residual state's two-qubit reduced density
    matrix (dominant eigenvector -> |00>), then removed from the residual.
    """
    batch = targets.shape[0]
    m_blocks = len(topology)
    out = np.empty((batch, m_blocks, 4, 4), dtype=targets.dtype)
    psi = targets.copy()
    for m in reversed(range(m_blocks)):
        rho = _reduced_density(psi, topology[m], n)
        _, vecs = np.linalg.eigh(rho)
        v = vecs[..., ::-1]
        out[:, m] = v
        psi = apply_block(psi, v, topology[m], n, adjoint=True)
    return out


# ------------------------------------------------------------------ optimiser

def sweep(targets, topology, unitaries, n=N_QUBITS):
    """One forward + one backward sweep of exact block updates (in place).

    Returns the squared fidelity after the sweep.
    """
    m_blocks = len(topology)
    batch = targets.shape[0]
    # Backward environments: back[m] = U_{m+1}^dag ... U_{M-1}^dag |T>.
    back = [None] * m_blocks
    psi = targets
    for m in reversed(range(m_blocks)):
        back[m] = psi
        psi = apply_block(psi, unitaries[:, m], topology[m], n, adjoint=True)
    phi = np.zeros((batch, 1 << n), dtype=targets.dtype)
    phi[:, 0] = 1
    fwd = [None] * m_blocks
    for m in range(m_blocks):
        fwd[m] = phi
        u, _ = _polar_update(_environment(phi, back[m], topology[m], n))
        unitaries[:, m] = u
        phi = apply_block(phi, u, topology[m], n)
    psi = targets
    overlap = None
    for m in reversed(range(m_blocks)):
        u, overlap = _polar_update(_environment(fwd[m], psi, topology[m], n))
        unitaries[:, m] = u
        psi = apply_block(psi, u, topology[m], n, adjoint=True)
    return overlap ** 2


@dataclass
class OptimResult:
    fidelity: np.ndarray        # (S,) best squared fidelity per target
    unitaries: np.ndarray       # (S, M, 4, 4) best blocks per target
    restart_fidelity: np.ndarray  # (S, R) final fidelity of every restart
    sweeps_run: int
    block_updates: int          # total single-block SVD updates (cost proxy)


def optimise(targets, topology, *, restarts=4, sweeps=200, init="mixed",
             warm_start=None, dtype=complex, tol=1e-10, patience=10,
             seed=0, n=N_QUBITS, noise=0.2):
    """Optimise all targets for one fixed topology.

    ``init`` in {"greedy", "haar", "identity", "mixed"}; "mixed" uses the
    greedy initialisation for restart 0 and near-identity/Haar random
    starts for the others. ``warm_start`` (S, M, 4, 4) replaces restart 0.
    Sweeps stop once no fidelity in the batch improves by more than ``tol``
    for ``patience`` consecutive sweeps.
    """
    topology = canonical_topology(topology)
    targets = np.asarray(targets).astype(dtype)
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    s, m_blocks = targets.shape[0], len(topology)
    rng = np.random.default_rng(seed)
    inits = []
    for r in range(restarts):
        if r == 0 and warm_start is not None:
            inits.append(np.asarray(warm_start).astype(dtype))
        elif init == "greedy" or (init == "mixed" and r == 0):
            inits.append(greedy_disentangle(targets, topology, n))
        elif init == "identity" or (init == "mixed" and r % 2 == 1):
            inits.append(near_identity_unitaries(rng, (s, m_blocks), noise, dtype))
        elif init in ("haar", "mixed"):
            inits.append(haar_unitaries(rng, (s, m_blocks), dtype))
        else:
            raise ValueError(f"Unknown init {init!r}")
    big_t = np.concatenate([targets] * restarts)
    u = np.concatenate(inits)
    prev = np.full(big_t.shape[0], -1.0)
    stale = 0
    done = 0
    for k in range(sweeps):
        f = sweep(big_t, topology, u, n)
        done = k + 1
        if np.max(f - prev) <= tol:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0
        prev = f
    f = fidelities(big_t, topology, u, n).reshape(restarts, s)
    best = np.argmax(f, axis=0)
    ub = u.reshape(restarts, s, m_blocks, 4, 4)[best, np.arange(s)]
    return OptimResult(fidelity=f[best, np.arange(s)], unitaries=ub,
                       restart_fidelity=f.T.copy(), sweeps_run=done,
                       block_updates=2 * done * m_blocks * s * restarts)


def is_unitary(u, atol=1e-10):
    u = np.asarray(u)
    eye = np.eye(u.shape[-1])
    return bool(np.allclose(np.conj(np.swapaxes(u, -1, -2)) @ u, eye, atol=atol))
