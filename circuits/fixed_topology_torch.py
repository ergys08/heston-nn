"""PyTorch engine for the fixed-topology SVD optimiser (search workhorse).

Same conventions and update rule as circuits.fixed_topology (the NumPy
reference implementation, validated against the saved AQCE circuits); this
version is ~3x faster on CPU and drops converged batch elements early.
Real dtype (float64) restricts blocks to orthogonal matrices, which is a
fast screening model for the real nonnegative Heston amplitudes; final
results are re-optimised with complex unitaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fixed_topology import N_QUBITS, canonical_topology, pair_permutation

_PERMS: dict = {}


def _perm(pair, n):
    key = (pair, n)
    if key not in _PERMS:
        _PERMS[key] = torch.from_numpy(pair_permutation(*pair, n))
    return _PERMS[key]


def apply_block(states, u, pair, n=N_QUBITS, adjoint=False):
    p = _perm(pair, n)
    x = states[:, p].view(states.shape[0], 4, -1)
    if adjoint:
        u = u.mH
    out = torch.empty_like(states)
    out[:, p] = (u @ x).reshape(states.shape[0], -1)
    return out


def prepare(topology, u, n=N_QUBITS):
    psi = torch.zeros(u.shape[0], 1 << n, dtype=u.dtype)
    psi[:, 0] = 1
    for m, pair in enumerate(topology):
        psi = apply_block(psi, u[:, m], pair, n)
    return psi


def fidelities(targets, topology, u, n=N_QUBITS):
    psi = prepare(topology, u, n)
    return (targets.conj() * psi).sum(-1).abs().square()


def _polar(env):
    w, s, vh = torch.linalg.svd(env)
    return vh.mH @ w.mH, s.sum(-1)


def sweep(targets, topology, u, n=N_QUBITS):
    """Forward + backward sweep, in place on u; returns fidelity."""
    m_blocks = len(topology)
    batch = targets.shape[0]
    back = [None] * m_blocks
    psi = targets
    for m in reversed(range(m_blocks)):
        back[m] = psi
        psi = apply_block(psi, u[:, m], topology[m], n, adjoint=True)
    phi = torch.zeros(batch, 1 << n, dtype=targets.dtype)
    phi[:, 0] = 1
    fwd = [None] * m_blocks
    for m in range(m_blocks):
        fwd[m] = phi
        p = _perm(topology[m], n)
        env = phi[:, p].view(batch, 4, -1) @ back[m][:, p].view(batch, 4, -1).mH
        new, _ = _polar(env)
        u[:, m] = new
        phi = apply_block(phi, new, topology[m], n)
    psi = targets
    overlap = None
    for m in reversed(range(m_blocks)):
        p = _perm(topology[m], n)
        env = fwd[m][:, p].view(batch, 4, -1) @ psi[:, p].view(batch, 4, -1).mH
        new, overlap = _polar(env)
        u[:, m] = new
        psi = apply_block(psi, new, topology[m], n, adjoint=True)
    return overlap.square()


def random_orthogonal(gen, shape, dtype):
    if dtype.is_complex:
        z = torch.complex(torch.randn(*shape, 4, 4, generator=gen, dtype=torch.float64),
                          torch.randn(*shape, 4, 4, generator=gen, dtype=torch.float64))
    else:
        z = torch.randn(*shape, 4, 4, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    return q * (d / d.abs()).unsqueeze(-2)


def near_identity(gen, shape, dtype, scale):
    a = torch.randn(*shape, 4, 4, generator=gen, dtype=torch.float64)
    if dtype.is_complex:
        a = torch.complex(a, torch.randn(*shape, 4, 4, generator=gen, dtype=torch.float64))
    a = (a - a.mH) / 2
    return torch.linalg.matrix_exp(scale * a)


def greedy_disentangle(targets, topology, n=N_QUBITS):
    batch, m_blocks = targets.shape[0], len(topology)
    out = torch.empty(batch, m_blocks, 4, 4, dtype=targets.dtype)
    psi = targets.clone()
    for m in reversed(range(m_blocks)):
        p = _perm(topology[m], n)
        g = psi[:, p].view(batch, 4, -1)
        _, vecs = torch.linalg.eigh(g @ g.mH)
        v = vecs.flip(-1)
        out[:, m] = v
        psi = apply_block(psi, v, topology[m], n, adjoint=True)
    return out


@dataclass
class OptimResult:
    fidelity: np.ndarray          # (S,) best fidelity per state
    unitaries: np.ndarray         # (S, M, 4, 4)
    restart_fidelity: np.ndarray  # (S, R)
    sweeps_run: int               # sweeps of the longest-running element
    block_updates: int            # total per-element block SVD updates


def optimise(targets, topology, *, restarts=4, sweeps=200, dtype=torch.float64,
             warm_start=None, tol=1e-9, window=10, seed=0, n=N_QUBITS,
             noise=0.3, init_mix=("identity", "haar"), stop_at=None,
             prescreen_sweeps=0, keep=None):
    """Batched optimisation of every (state, restart) pair.

    An element is frozen once its fidelity improved by less than ``tol``
    over the last ``window`` sweeps (it then costs nothing further).
    Restart 0 uses the deterministic greedy (AQCE-like) initialisation, or
    ``warm_start`` if given; restart r >= 1 uses init_mix[(r-1) % len].
    With ``stop_at``, elements reaching that fidelity are also frozen
    (cheap screening: only "does it pass" matters).

    Successive halving: with ``prescreen_sweeps`` > 0 and ``keep`` < restarts,
    all restarts first run ``prescreen_sweeps`` sweeps; then only the best
    ``keep`` restarts of each state continue (up to ``sweeps`` in total).
    """
    topology = canonical_topology(topology)
    t = torch.as_tensor(np.asarray(targets)).to(dtype)
    t = t / torch.linalg.vector_norm(t, dim=1, keepdim=True)
    s, m_blocks = t.shape[0], len(topology)
    gen = torch.Generator().manual_seed(seed)
    inits = []
    for r in range(restarts):
        kind = "greedy" if r == 0 else init_mix[(r - 1) % len(init_mix)]
        if r == 0 and warm_start is not None:
            inits.append(torch.as_tensor(np.asarray(warm_start)).to(dtype))
        elif kind == "greedy":
            inits.append(greedy_disentangle(t, topology, n))
        elif kind == "identity":
            inits.append(near_identity(gen, (s, m_blocks), dtype, noise))
        elif kind == "haar":
            inits.append(random_orthogonal(gen, (s, m_blocks), dtype))
        else:
            raise ValueError(kind)
    big_t = t.repeat(restarts, 1)
    u = torch.cat(inits)
    total = big_t.shape[0]
    final_f = torch.zeros(total, dtype=torch.float64)
    active = torch.arange(total)
    history = []
    updates = 0
    done = 0
    for k in range(sweeps):
        u_active = u[active]
        f = sweep(big_t[active], topology, u_active, n)
        u[active] = u_active
        final_f[active] = f.real.to(torch.float64) if f.is_complex() else f
        updates += 2 * m_blocks * len(active)
        done = k + 1
        history.append(final_f.clone())
        keep_mask = torch.ones(len(active), dtype=torch.bool)
        if k + 1 == prescreen_sweeps and keep is not None and keep < restarts:
            # rank restarts per state; drop all but the best `keep`
            rank = torch.argsort(torch.argsort(-final_f.reshape(restarts, s), dim=0), dim=0)
            survivors = (rank < keep).reshape(-1)
            keep_mask &= survivors[active]
        if len(history) > window and k + 1 > prescreen_sweeps:
            keep_mask &= (final_f - history[-window - 1])[active] > tol
        if stop_at is not None:
            # A state whose best restart passed needs no further work.
            passed = (final_f.reshape(restarts, s) >= stop_at).any(0)
            keep_mask &= ~passed[active % s]
        active = active[keep_mask]
        if len(active) == 0:
            break
    f = fidelities(big_t, topology, u, n).to(torch.float64).reshape(restarts, s)
    best = torch.argmax(f, dim=0)
    ub = u.reshape(restarts, s, m_blocks, 4, 4)[best, torch.arange(s)]
    return OptimResult(fidelity=f[best, torch.arange(s)].numpy(),
                       unitaries=ub.numpy(), restart_fidelity=f.T.numpy().copy(),
                       sweeps_run=done, block_updates=updates)
