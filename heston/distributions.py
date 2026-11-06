"""Heston terminal log-return PDFs and probability-bin discretisation.

No file I/O. The functions return NumPy arrays and validate their outputs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParameters:
    v0: float
    kappa: float
    theta: float
    xi: float       # Volatility of variance (called sigma in the source CSV).
    rho: float

    def __post_init__(self):
        positive = (self.v0, self.kappa, self.theta, self.xi)
        if not np.all(np.isfinite((*positive, self.rho))):
            raise ValueError("Heston parameters must be finite")
        if min(positive) <= 0 or not -1 < self.rho < 1:
            raise ValueError("Require v0, kappa, theta, xi > 0 and -1 < rho < 1")


def characteristic_function(u, p, maturity=1.0, drift=0.05):
    """E[exp(i*u*log(S_T/S_0))] under the Heston model."""
    if not np.isfinite(maturity) or maturity <= 0 or not np.isfinite(drift):
        raise ValueError("Maturity must be positive and drift must be finite")
    u = np.asarray(u, dtype=complex)
    b = p.kappa - p.rho * p.xi * 1j * u
    d = np.sqrt(b**2 + p.xi**2 * (u**2 + 1j * u))
    g = (b - d) / (b + d)
    decay = np.exp(-d * maturity)
    C = (p.kappa * p.theta / p.xi**2) * (
        (b - d) * maturity - 2 * np.log((1 - g * decay) / (1 - g))
    )
    D = ((b - d) / p.xi**2) * (1 - decay) / (1 - g * decay)
    return np.exp(1j * u * drift * maturity + C + p.v0 * D)


def heston_pdf(p, *, n_fft=16384, halfwidth=3.5, maturity=1.0, drift=0.05):
    """Return (log_return_grid, density, diagnostics) via Fourier inversion.

    The Fourier grid is periodic; boundary and negative-mass checks catch
    many (but not all) aliasing and precision problems.
    """
    if n_fft < 256 or n_fft & (n_fft - 1):
        raise ValueError("n_fft must be a power of two >= 256")
    if not np.isfinite(halfwidth) or halfwidth <= 0:
        raise ValueError("halfwidth must be positive and finite")

    dx = 2 * halfwidth / n_fft
    x = -halfwidth + dx * np.arange(n_fft)
    frequencies = 2 * np.pi * np.fft.fftfreq(n_fft, d=dx)
    cf = characteristic_function(frequencies, p, maturity, drift)
    raw_pdf = np.fft.fft(cf * np.exp(1j * frequencies * halfwidth)).real / (n_fft * dx)

    if not np.all(np.isfinite(raw_pdf)):
        raise ArithmeticError("Non-finite density from Fourier inversion")
    negative_mass = float(np.maximum(-raw_pdf, 0).sum() * dx)
    if negative_mass > 1e-8:
        raise ArithmeticError(f"Negative density mass too large: {negative_mass:.3g}")

    pdf = np.maximum(raw_pdf, 0)
    raw_integral = float(pdf.sum() * dx)
    if not 0.99 < raw_integral < 1.01:
        raise ArithmeticError(f"Invalid PDF integral: {raw_integral:.8g}")
    pdf /= raw_integral

    outer = max(1, n_fft // 20)
    edge_mass = float((pdf[:outer].sum() + pdf[-outer:].sum()) * dx)
    if edge_mass > 1e-5:
        raise ArithmeticError(f"FFT grid boundary mass too large: {edge_mass:.3g}")

    # Built-in sanity check: phi(0)=1 and the resulting density integrates to 1.
    if abs(characteristic_function(0, p, maturity, drift) - 1) > 1e-10:
        raise ArithmeticError("Characteristic function fails phi(0)=1")
    if not np.isclose(pdf.sum() * dx, 1, atol=1e-10):
        raise ArithmeticError("PDF failed normalisation")
    return x, pdf, {"negative_mass": negative_mass, "edge_mass": edge_mass}


def discretize_pdf(x, pdf, *, n_qubits=8, cutoff=2.5):
    """Return (edges, bin_masses, conditional_probabilities, retained_mass).

    Each of 2**n_qubits bins receives its integral probability. The conditional
    probabilities sum to one; the retained mass records the discarded tails.
    """
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or not 1 <= n_qubits <= 16:
        raise ValueError("n_qubits must be an integer between 1 and 16")
    x, pdf = np.asarray(x), np.asarray(pdf)
    if x.ndim != 1 or len(x) < 2 or x.shape != pdf.shape:
        raise ValueError("x and pdf must have matching one-dimensional shapes")
    dx = x[1] - x[0]
    if not np.isfinite(dx) or dx <= 0 or not np.allclose(np.diff(x), dx):
        raise ValueError("x must be a uniformly spaced increasing grid")
    if not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
        raise ValueError("pdf must be finite and nonnegative")
    if not 0 < cutoff < min(-x[0], x[-1] + dx):
        raise ValueError("cutoff is outside the FFT grid")

    edges = np.linspace(-cutoff, cutoff, 2**n_qubits + 1)
    # Periodic endpoint closes the FFT density grid. Its CDF is integrated
    # by trapezoids, then interpolated at all bin boundaries.
    x_extended = np.r_[x, x[-1] + dx]
    pdf_extended = np.r_[pdf, pdf[0]]
    cdf = np.r_[0.0, np.cumsum((pdf_extended[:-1] + pdf_extended[1:]) * dx / 2)]
    if not np.isclose(cdf[-1], 1.0, atol=1e-5):
        raise ArithmeticError("Input density does not integrate to one")
    cdf /= cdf[-1]
    masses = np.diff(np.interp(edges, x_extended, cdf))
    if np.min(masses) < -1e-12:
        raise ArithmeticError("Negative discretised bin probability")
    masses = np.maximum(masses, 0)
    retained = float(masses.sum())
    if not 0 < retained <= 1 + 1e-9:
        raise ArithmeticError("Invalid retained probability mass")
    probabilities = masses / retained

    # Built-in checks: exactly 2**n_qubits bins and normalised probabilities.
    if len(probabilities) != 2**n_qubits or not np.isclose(probabilities.sum(), 1, atol=1e-10):
        raise ArithmeticError("Invalid discretisation")
    return edges, masses, probabilities, retained
