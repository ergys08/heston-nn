"""Dimensionless Heston coordinates and numerically checked amplitude states."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import expm
from heston.distributions import HestonParameters, heston_pdf, discretize_pdf


@dataclass(frozen=True)
class Coordinates:
    a: float   # kappa*T
    b: float   # xi^2/(2*kappa*theta)
    c: float   # v0/theta
    d: float   # theta*T
    rho: float

    def parameters(self, maturity=1.0):
        if not np.all(np.isfinite([self.a, self.b, self.c, self.d, self.rho, maturity])):
            raise ValueError("Coordinates must be finite")
        if min(self.a, self.b, self.c, self.d, maturity) <= 0 or abs(self.rho) >= 1:
            raise ValueError("Require positive a,b,c,d,T and |rho| < 1")
        kappa = self.a / maturity
        theta = self.d / maturity
        return HestonParameters(self.c * theta, kappa, theta,
                                math.sqrt(2 * kappa * theta * self.b), self.rho)


def moments(p, T=1.0):
    """Exact first and second moments via a closed linear moment system.

    X has drift -v/2 (constant asset drift set to zero). Initial variance
    is deterministic. Also returns E[A] and Var[A] for integrated variance.
    """
    k, theta, xi, rho = p.kappa, p.theta, p.xi, p.rho
    # State [1, E[v], Var[v], Cov[X,v], Var[X], Cov[A,v], Var[A]].
    M = np.zeros((7, 7))
    M[1, 0], M[1, 1] = k * theta, -k
    M[2, 1], M[2, 2] = xi**2, -2*k
    M[3, 1], M[3, 2], M[3, 3] = rho*xi, -.5, -k
    M[4, 1], M[4, 3] = 1., -1.
    M[5, 2], M[5, 5] = 1., -k
    M[6, 5] = 2.
    z = expm(T*M) @ np.array([1., p.v0, 0., 0., 0., 0., 0.])
    EA = theta*T + (p.v0-theta)*(-math.expm1(-k*T))/k
    if not np.all(np.isfinite(z)) or z[4] <= 0 or z[6] < -1e-12:
        raise ArithmeticError("Invalid analytic moments")
    return -.5*EA, math.sqrt(z[4]), EA, max(0., z[6])


def distance(p, q):
    """Squared-overlap infidelity, evaluated stably for nearby probabilities."""
    x, y = np.sqrt(p), np.sqrt(q)
    return float(.5 * np.sum((x-y)**2) * (1 + np.dot(x, y)))


def make_state(coords, *, n_qubits=8, maturity=1., cutoff=6.,
               n_fft=16384, max_fft=1048576, numerical_tol=1e-8,
               moment_tol=2e-4):
    """Return conditional bin probabilities and diagnostics.

    Uses the repo's Fourier inversion and bin-integral routines. Compare
    increased resolution AND doubled domain at equal fine spacing. Check
    the unconditioned numerical mean/variance against independent exact
    moments. Fail explicitly if the configured numerical budget is exhausted.
    These checks are numerical convergence tests, not rigorous error bounds.
    """
    p = coords.parameters(maturity)
    mean, std, EA, VA = moments(p, maturity)
    halfwidth = abs(mean) + max(12., 2*cutoff+2) * std

    def invert(n, width):
        x, pdf, diag = heston_pdf(p, n_fft=n, halfwidth=width,
                                 maturity=maturity, drift=0.)
        z = (x-mean)/std
        f = pdf*std
        dz = z[1]-z[0]
        m1 = float(np.sum(z*f)*dz)
        m2 = float(np.sum(z*z*f)*dz)
        # Repository discretizer integrates over [-cutoff,cutoff].
        _, _, prob, retained = discretize_pdf(z, f, n_qubits=n_qubits, cutoff=cutoff)
        return prob, retained, m1, m2, diag

    n = n_fft
    last = "No inversion attempted"
    while 4*n <= max_fft:
        try:
            coarse = invert(n, halfwidth)
            fine = invert(2*n, halfwidth)
            wide = invert(4*n, 2*halfwidth)
        except ArithmeticError as exc:
            last = str(exc)
            # Boundary mass signals a narrow domain; preserve resolution.
            if "boundary" in last.lower():
                halfwidth *= 2
            n *= 2
            continue
        resolution = distance(coarse[0], fine[0])
        domain = distance(fine[0], wide[0])
        mass_error = max(abs(coarse[1]-fine[1]), abs(fine[1]-wide[1]))
        moment_error = max(abs(fine[2]), abs(fine[3]-1),
                           abs(wide[2]), abs(wide[3]-1))
        if (max(resolution, domain) <= numerical_tol and
                mass_error <= math.sqrt(numerical_tol) and moment_error <= moment_tol):
            prob, retained, m1, m2, diag = wide
            return prob, dict(mean=mean, std=std, integrated_variance_mean=EA,
                              integrated_variance_cv2=VA/EA**2,
                              retained_mass=retained, fft_points=4*n,
                              fft_halfwidth=2*halfwidth,
                              resolution_infidelity=resolution,
                              domain_infidelity=domain,
                              retained_mass_difference=mass_error,
                              moment_error=moment_error, fft_mean_z=m1,
                              fft_second_moment_z=m2, **diag)
        last = (f"resolution={resolution:.3g}, domain={domain:.3g}, "
                f"mass={mass_error:.3g}, moment={moment_error:.3g}")
        if domain > numerical_tol or abs(fine[2]-wide[2]) > moment_tol/2 or abs(fine[3]-wide[3]) > moment_tol/2:
            halfwidth *= 2
        n *= 2
    raise ArithmeticError(f"FFT convergence budget exhausted: {last}")
