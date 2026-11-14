"""Product-formula ansatz  |phi(theta)> = prod_k exp(-i theta_k P_k) |0...0>
with analytic gradients, and fitting with shared ("fixed") plus per-state
("variable") coefficients.

The product (first-order Trotter, one step) is exactly what a compiler
emits for exp(-iH); its coefficients are optimised directly so no Trotter
error is incurred on top of the fit.  `exact_exponential_fidelity` evaluates
the exact exp(-iH)|0> for comparison.

Loss: mean infidelity over the batch of targets (optionally the mean of
-log F, which weights poor states slightly more).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import minimize

from paulis import DIM, generator

E0 = np.zeros(DIM); E0[0] = 1.0


class ProductAnsatz:
    def __init__(self, labels):
        self.labels = list(labels)
        self.gens = [generator(l) for l in self.labels]
        self.K = len(self.labels)

    def states(self, theta: np.ndarray) -> np.ndarray:
        """theta (J,K) -> output states (J,256)."""
        theta = np.atleast_2d(theta)
        s = np.tile(E0, (theta.shape[0], 1))
        for k, g in enumerate(self.gens):
            c, sn = np.cos(theta[:, k])[:, None], np.sin(theta[:, k])[:, None]
            s = c * s + sn * g.apply(s)
        return s

    def fidelity_grad(self, theta: np.ndarray, targets: np.ndarray):
        """Return F (J,), dF/dtheta (J,K) for real targets (J,256)."""
        theta = np.atleast_2d(theta)
        J = theta.shape[0]
        cos, sin = np.cos(theta), np.sin(theta)
        fwd = np.empty((self.K + 1, J, DIM))
        fwd[0] = E0
        for k, g in enumerate(self.gens):
            fwd[k + 1] = cos[:, k, None] * fwd[k] + sin[:, k, None] * g.apply(fwd[k])
        phi = fwd[self.K]
        overlap = np.einsum("jd,jd->j", targets, phi)
        F = overlap ** 2
        grad = np.empty((J, self.K))
        lam = targets.copy()
        for k in range(self.K - 1, -1, -1):
            g = self.gens[k]
            grad[:, k] = np.einsum("jd,jd->j", lam, g.apply(fwd[k + 1]))
            lam = cos[:, k, None] * lam - sin[:, k, None] * g.apply(lam)
        grad *= 2 * overlap[:, None]
        return F, grad


@dataclass
class FitResult:
    labels: list
    variable: list            # POSITIONS (0-based, in label order) whose coefficient varies per state
    theta_fixed: np.ndarray   # (K_f,) shared coefficients, in order of the fixed positions
    theta_var: np.ndarray     # (J, K_v)
    fidelities: np.ndarray    # (J,)
    loss: float
    n_iter: int
    seconds: float
    restarts: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def mean_fidelity(self): return float(self.fidelities.mean())
    @property
    def min_fidelity(self): return float(self.fidelities.min())
    @property
    def n_variable(self): return len(self.variable)
    @property
    def variable_labels(self): return [self.labels[i] for i in self.variable]
    @property
    def fixed_positions(self): return [i for i in range(len(self.labels)) if i not in set(self.variable)]

    def theta_full(self) -> np.ndarray:
        """(J,K) coefficient matrix in label order."""
        J = self.theta_var.shape[0]
        theta = np.zeros((J, len(self.labels)))
        theta[:, self.fixed_positions] = self.theta_fixed
        if self.variable:
            theta[:, list(self.variable)] = self.theta_var
        return theta

    def coefficients_dict(self):
        return {"fixed": {f"{i}:{self.labels[i]}": float(v) for i, v in zip(self.fixed_positions, self.theta_fixed)},
                "variable": {f"{i}:{self.labels[i]}": self.theta_var[:, k].tolist() for k, i in enumerate(self.variable)}}


def resolve_variable(labels, variable):
    """Variable spec -> sorted list of positions.  Ints are positions; strings
    select EVERY position carrying that label."""
    out = set()
    for v in variable:
        if isinstance(v, (int, np.integer)):
            out.add(int(v))
        else:
            out.update(i for i, l in enumerate(labels) if l == v)
    return sorted(out)


def _loss_and_grad(F, grad, kind):
    if kind == "mean":
        return float(np.mean(1 - F)), -grad / len(F)
    if kind == "log":
        Fc = np.maximum(F, 1e-12)
        return float(np.mean(-np.log(Fc))), -grad / (Fc[:, None] * len(F))
    raise ValueError(kind)


def fit(labels, targets, variable=(), *, init_fixed=None, init_var=None,
        loss="mean", maxiter=2000, gtol=1e-9, restarts=1, seed=0,
        init_scale=0.3, perturb=0.1, verbose=False) -> FitResult:
    """Optimise shared + per-state coefficients.

    `variable` lists positions (ints) or labels (all positions with that label).
    Several restarts: restart 0 uses the supplied initial values (or zeros),
    odd restarts perturb them, even restarts draw random angles; best loss wins.
    """
    labels = list(labels)
    var_idx = np.array(resolve_variable(labels, variable), dtype=int)
    targets = np.atleast_2d(targets)
    J = targets.shape[0]
    ansatz = ProductAnsatz(labels)
    fixed_idx = np.array([i for i in range(len(labels)) if i not in set(var_idx.tolist())], dtype=int)
    Kf, Kv = len(fixed_idx), len(var_idx)
    rng = np.random.default_rng(seed)

    def unpack(x):
        theta = np.zeros((J, len(labels)))
        theta[:, fixed_idx] = x[:Kf]
        if Kv:
            theta[:, var_idx] = x[Kf:].reshape(J, Kv)
        return theta

    def objective(x):
        theta = unpack(x)
        F, grad = ansatz.fidelity_grad(theta, targets)
        L, dL = _loss_and_grad(F, grad, loss)
        gx = np.concatenate([dL[:, fixed_idx].sum(0), dL[:, var_idx].ravel()])
        return L, gx

    base_fixed = np.zeros(Kf) if init_fixed is None else np.asarray(init_fixed, float)
    base_var = np.zeros((J, Kv)) if init_var is None else np.asarray(init_var, float)
    if base_var.ndim == 1:
        base_var = np.tile(base_var, (J, 1))
    assert base_fixed.shape == (Kf,) and base_var.shape == (J, Kv)
    best = None
    t0 = time.time()
    for r in range(restarts):
        if r == 0:
            x0 = np.concatenate([base_fixed, base_var.ravel()])
        elif r % 2 == 1:
            x0 = np.concatenate([base_fixed, base_var.ravel()]) + perturb * rng.normal(size=Kf + J * Kv)
        else:
            x0 = init_scale * rng.normal(size=Kf + J * Kv)
        res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15, "maxcor": 30})
        if verbose:
            print(f"  restart {r}: loss {res.fun:.3e} iters {res.nit}")
        if best is None or res.fun < best.fun:
            best = res
    theta = unpack(best.x)
    F = ansatz.fidelity_grad(theta, targets)[0]
    return FitResult(labels, var_idx.tolist(), best.x[:Kf].copy(), best.x[Kf:].reshape(J, Kv).copy(),
                     F, float(best.fun), int(best.nit), time.time() - t0, restarts)


def refit_variable_only(labels, variable, theta_fixed, targets, *, init_var=None, maxiter=1000,
                        restarts=3, seed=0) -> FitResult:
    """Held-out evaluation: freeze shared coefficients, optimise only the
    per-state variable coefficients (each state is independent)."""
    labels = list(labels)
    var_idx = np.array(resolve_variable(labels, variable), dtype=int)
    targets = np.atleast_2d(targets)
    J = targets.shape[0]
    ansatz = ProductAnsatz(labels)
    fixed_idx = np.array([i for i in range(len(labels)) if i not in set(var_idx.tolist())], dtype=int)
    Kv = len(var_idx)
    theta = np.zeros((J, len(labels)))
    theta[:, fixed_idx] = theta_fixed
    if Kv == 0:
        F = ansatz.fidelity_grad(theta, targets)[0]
        return FitResult(labels, [], np.asarray(theta_fixed), np.zeros((J, 0)), F, float(np.mean(1 - F)), 0, 0.0)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def objective(x):
        th = theta.copy(); th[:, var_idx] = x.reshape(J, Kv)
        F, grad = ansatz.fidelity_grad(th, targets)
        return float(np.sum(1 - F)), -grad[:, var_idx].ravel()

    starts = [np.zeros((J, Kv)) if init_var is None else np.asarray(init_var, float)]
    if np.ndim(starts[0]) == 1:
        starts = [np.tile(starts[0], (J, 1))]
    for r in range(1, restarts):
        starts.append(starts[0] + 0.3 * rng.normal(size=(J, Kv)))
    best_x = None; best_F = None
    for x0 in starts:
        res = minimize(objective, x0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-15})
        th = theta.copy(); th[:, var_idx] = res.x.reshape(J, Kv)
        F = ansatz.fidelity_grad(th, targets)[0]
        if best_F is None:
            best_F, best_x = F, res.x.reshape(J, Kv)
        else:
            better = F > best_F
            best_F = np.where(better, F, best_F)
            best_x[better] = res.x.reshape(J, Kv)[better]
    return FitResult(labels, var_idx.tolist(), np.asarray(theta_fixed), best_x, best_F, float(np.mean(1 - best_F)), 0,
                     time.time() - t0, restarts)


def fit_lowrank(labels, targets, rank, *, init_theta0=None, init_dirs=None, init_tau=None,
                maxiter=3000, restarts=1, seed=0, loss="mean") -> dict:
    """theta_j = theta0 + sum_m tau_{jm} d_m : `rank` independent scalars per
    state, each of which may move several rotation angles at once."""
    labels = list(labels); K = len(labels)
    targets = np.atleast_2d(targets); J = targets.shape[0]
    ansatz = ProductAnsatz(labels)
    rng = np.random.default_rng(seed)
    M = rank

    def unpack(x):
        th0 = x[:K]; D = x[K:K + M * K].reshape(M, K); tau = x[K + M * K:].reshape(J, M)
        return th0, D, tau

    def objective(x):
        th0, D, tau = unpack(x)
        theta = th0[None, :] + tau @ D
        F, grad = ansatz.fidelity_grad(theta, targets)
        L, dL = _loss_and_grad(F, grad, loss)
        g0 = dL.sum(0); gD = tau.T @ dL; gtau = dL @ D.T
        # tiny penalty keeps the tau/D scale ambiguity from drifting
        L += 1e-6 * np.sum(D ** 2); gD += 2e-6 * D
        return L, np.concatenate([g0, gD.ravel(), gtau.ravel()])

    th0 = np.zeros(K) if init_theta0 is None else np.asarray(init_theta0, float)
    D0 = (0.1 * rng.normal(size=(M, K))) if init_dirs is None else np.asarray(init_dirs, float)
    tau0 = np.zeros((J, M)) if init_tau is None else np.asarray(init_tau, float)
    best = None
    for r in range(restarts):
        x0 = np.concatenate([th0, D0.ravel(), tau0.ravel()])
        if r > 0:
            x0 = x0 + 0.1 * rng.normal(size=x0.size)
        res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-15, "maxcor": 30})
        if best is None or res.fun < best.fun:
            best = res
    th0, D, tau = unpack(best.x)
    # normalise directions
    norms = np.linalg.norm(D, axis=1, keepdims=True); D = D / norms; tau = tau * norms.T
    F = ansatz.fidelity_grad(th0[None, :] + tau @ D, targets)[0]
    return {"labels": labels, "theta0": th0, "dirs": D, "tau": tau, "fidelities": F,
            "mean_fidelity": float(F.mean()), "min_fidelity": float(F.min()), "n_iter": int(best.nit)}


def lowrank_refit_tau(labels, theta0, dirs, targets, *, restarts=5, seed=0):
    """Held-out: optimise only the per-state tau (independent per state)."""
    ansatz = ProductAnsatz(labels)
    targets = np.atleast_2d(targets); J = targets.shape[0]; M = dirs.shape[0]
    rng = np.random.default_rng(seed)

    def objective(x):
        tau = x.reshape(J, M)
        F, grad = ansatz.fidelity_grad(theta0[None, :] + tau @ dirs, targets)
        return float(np.sum(1 - F)), -(grad @ dirs.T).ravel()

    best_F = None; best_tau = None
    for r in range(restarts):
        x0 = np.zeros(J * M) if r == 0 else 0.5 * rng.normal(size=J * M)
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-10, "ftol": 1e-15})
        tau = res.x.reshape(J, M)
        F = ansatz.fidelity_grad(theta0[None, :] + tau @ dirs, targets)[0]
        if best_F is None:
            best_F, best_tau = F, tau
        else:
            better = F > best_F; best_F = np.where(better, F, best_F); best_tau[better] = tau[better]
    return best_F, best_tau


def exact_exponential_states(labels, theta: np.ndarray) -> np.ndarray:
    """exp(sum_k theta_k A_k) e0 for each row of theta (J,K) using torch.matrix_exp."""
    import torch
    from paulis import dense_generator
    A = torch.tensor(np.stack([dense_generator(l) for l in labels]), dtype=torch.float64)
    th = torch.tensor(np.atleast_2d(theta), dtype=torch.float64)
    out = []
    for start in range(0, th.shape[0], 32):
        G = torch.einsum("jk,kab->jab", th[start:start + 32], A)
        out.append(torch.linalg.matrix_exp(G)[:, :, 0].numpy())
    return np.concatenate(out, 0)


def fit_exact_exponential(labels, targets, variable=(), *, init_fixed=None, init_var=None,
                          maxiter=300, lr=0.05) -> FitResult:
    """Fit the EXACT exponential exp(-i H_j)|0> with shared + per-state
    coefficients, using torch autograd through matrix_exp (slower)."""
    import torch
    from paulis import dense_generator
    labels = list(labels); var_idx = resolve_variable(labels, variable)
    targets = np.atleast_2d(targets); J = targets.shape[0]
    fixed_idx = [i for i in range(len(labels)) if i not in set(var_idx)]
    A = torch.tensor(np.stack([dense_generator(l) for l in labels]), dtype=torch.float64)
    T = torch.tensor(targets, dtype=torch.float64)
    tf = torch.tensor(np.zeros(len(fixed_idx)) if init_fixed is None else np.asarray(init_fixed, float),
                      dtype=torch.float64, requires_grad=True)
    tv0 = np.zeros((J, len(var_idx))) if init_var is None else np.asarray(init_var, float)
    tv = torch.tensor(tv0, dtype=torch.float64, requires_grad=True)
    params = [tf] + ([tv] if var_idx else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=maxiter, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=30, line_search_fn="strong_wolfe")
    t0 = time.time()

    def theta_full():
        th = torch.zeros(J, len(labels), dtype=torch.float64)
        th[:, fixed_idx] = tf
        if var_idx:
            th[:, var_idx] = tv
        return th

    def fidelities():
        G = torch.einsum("jk,kab->jab", theta_full(), A)
        phi = torch.linalg.matrix_exp(G)[:, :, 0]
        return (T * phi).sum(1) ** 2

    def closure():
        opt.zero_grad()
        L = (1 - fidelities()).mean()
        L.backward()
        return L
    opt.step(closure)
    F = fidelities().detach().numpy()
    return FitResult(labels, var_idx, tf.detach().numpy().copy(), tv.detach().numpy().copy().reshape(J, len(var_idx)),
                     F, float(np.mean(1 - F)), maxiter, time.time() - t0)


if __name__ == "__main__":
    from scipy.linalg import expm
    from paulis import dense_generator
    rng = np.random.default_rng(1)
    labels = ["IIYIIIII", "IIYZIIII", "IZYIIIII", "YXIIIIII", "IIIIYZZI", "IIIIIIIY", "IYIIIIII"]
    ans = ProductAnsatz(labels)
    theta = rng.normal(size=(3, len(labels)))
    # product vs explicit matrix product
    for j in range(3):
        U = np.eye(DIM)
        for k, l in enumerate(labels):
            U = expm(theta[j, k] * dense_generator(l)) @ U
        assert np.allclose(U[:, 0], ans.states(theta)[j])
    targets = rng.normal(size=(3, DIM)); targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    F, g = ans.fidelity_grad(theta, targets)
    # finite-difference gradient check
    eps = 1e-6
    for k in range(len(labels)):
        tp = theta.copy(); tp[:, k] += eps
        tm = theta.copy(); tm[:, k] -= eps
        Fp = (np.einsum("jd,jd->j", targets, ans.states(tp)))**2
        Fm = (np.einsum("jd,jd->j", targets, ans.states(tm)))**2
        assert np.allclose((Fp - Fm) / (2 * eps), g[:, k], atol=1e-6), k
    # exact exponential agrees with expm
    ex = exact_exponential_states(labels, theta)
    for j in range(3):
        G = sum(theta[j, k] * dense_generator(l) for k, l in enumerate(labels))
        assert np.allclose(ex[j], expm(G)[:, 0])
    # timing on a realistic size
    from states import load_states
    _, S, _ = load_states(2013)
    from paulis import pair_block_strings
    labs = sum([pair_block_strings(i, i + 1) for i in range(7)], [])
    ans = ProductAnsatz(labs)
    th = 0.1 * rng.normal(size=(60, len(labs)))
    t = time.time()
    for _ in range(20):
        ans.fidelity_grad(th, S[:60])
    print(f"K={len(labs)} J=60: {(time.time()-t)/20*1e3:.2f} ms per fidelity+gradient")
    t = time.time()
    r = fit(labs, S[:8], variable=[], restarts=1, maxiter=500)
    print(f"fit K={len(labs)} J=8 fixed-only: mean F {r.mean_fidelity:.5f} min {r.min_fidelity:.5f} iters {r.n_iter} {r.seconds:.1f}s")
    r = fit(labs, S[:8], variable=labs, restarts=1, maxiter=500)
    print(f"fit K={len(labs)} J=8 all-variable: mean F {r.mean_fidelity:.5f} min {r.min_fidelity:.5f} iters {r.n_iter} {r.seconds:.1f}s")
    assert r.variable == list(range(len(labs))) and r.theta_var.shape == (8, len(labs))
    r2 = fit(labs, S[:8], variable=[3, 9], restarts=1, maxiter=50)
    assert r2.variable == [3, 9] and r2.theta_fixed.shape == (len(labs) - 2,)
    assert np.allclose(r2.theta_full()[:, [3, 9]], r2.theta_var)
    print("ansatz.py self-test passed")
