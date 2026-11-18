"""MPS bond-dimension sweep for the published 2013 Heston calibrations.

Run from the repository root::

    python -m MPS.bonds_changing_qubits_2013

Writes MPS/MPS_bonds_changing_qubits_2013.csv (250 dates x 5 resolutions).

Important: A dense 96-qubit amplitude vector is impossible to store. The 8-
and 12-qubit entries use the actual discretized Heston target. Larger-
resolution entries use a *controlled, explicitly labelled* 12/14/16-qubit
quantics approximation: the remaining least-significant qubits are in |+>.
The reference is selected using a continuous-limit fidelity bound for the
PDF as represented on the Fourier grid. These rows are constructive estimates
/ sufficient bond dimensions, NOT exact minimal 24/48/96-qubit ranks.

Bond dimension means the smallest maximum bond tested by the repository's
TT-SVD (``decompose_state``) reaching a specified state fidelity. This is
not a proof of the globally smallest bond dimension among arbitrary MPS.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from heston.build_distributions import read_calibrations
from heston.distributions import discretize_pdf, heston_pdf
from MPS.mps_decomposition import decompose_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "heston" / "heston_params_2013.csv"
DEFAULT_OUTPUT = ROOT / "MPS" / "MPS_bonds_changing_qubits_2013.csv"
QUBITS = (8, 12, 24, 48, 96)
TARGET_FIDELITIES = (0.99, 0.995, 0.999, 0.9999)


def _sqrt_linear_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact average of sqrt(linearly interpolated PDF) on an interval."""
    a, b = np.maximum(np.asarray(a), 0.0), np.maximum(np.asarray(b), 0.0)
    sa, sb = np.sqrt(a), np.sqrt(b)
    numerator = a + np.sqrt(a * b) + b
    return np.divide((2.0 / 3.0) * numerator, sa + sb,
                     out=np.zeros_like(numerator), where=(sa + sb) > 0)


def _sqrt_pdf_integrals(x: np.ndarray, pdf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Integrate sqrt(pdf) over dyadic bins, exactly for piecewise-linear PDF.

    This is consistent with discretize_pdf(), which integrates the PDF by
    trapezoids and CDF interpolation on the same Fourier grid.
    """
    dx = float(x[1] - x[0])
    xx = np.r_[x, x[-1] + dx]
    pp = np.r_[pdf, pdf[0]]
    whole = dx * _sqrt_linear_mean(pp[:-1], pp[1:])
    prefix = np.r_[0.0, np.cumsum(whole)]
    indices = np.searchsorted(xx, edges, side="right") - 1
    indices = np.clip(indices, 0, len(pp) - 2)
    widths = edges - xx[indices]
    fractions = np.clip(widths / dx, 0.0, 1.0)
    at_edges = pp[indices] + fractions * (pp[indices + 1] - pp[indices])
    antiderivative = prefix[indices] + widths * _sqrt_linear_mean(pp[indices], at_edges)
    integrals = np.diff(antiderivative)
    if np.min(integrals) < -1e-11:
        raise ArithmeticError("Negative integral of sqrt(pdf) from floating point error")
    return np.maximum(integrals, 0.0)


def continuous_lift_fidelity(x: np.ndarray, pdf: np.ndarray,
                             edges: np.ndarray, probabilities: np.ndarray,
                             retained_mass: float) -> float:
    """Fidelity of a coarse-bin uniform lift against the continuum target.

    For each n >= m, the m-bit state tensored with |+>**(n-m) has fidelity
    >= this continuous-limit value against the exactly binned n-qubit PDF.
    This bound is relative to the numerically reconstructed Heston density;
    the PDF inversion and floating-point integration introduce their own error.
    """
    h = float(edges[1] - edges[0])
    integrals = _sqrt_pdf_integrals(x, pdf, edges)
    overlap = float(np.sum(np.sqrt(probabilities / h) * integrals) /
                    math.sqrt(retained_mass))
    if not np.isfinite(overlap) or overlap < -1e-10 or overlap > 1 + 1e-5:
        raise ArithmeticError(f"Invalid continuous overlap: {overlap}")
    return float(np.clip(overlap, 0.0, 1.0) ** 2)


def _tag(fidelity: float) -> str:
    return str(fidelity).replace("0.", "")


def min_ttsvd_bond(amplitudes: np.ndarray, fidelity: float,
                   cache: dict[int, float]) -> tuple[int, float]:
    """Smallest bond whose *greedy TT-SVD* output meets this fidelity.

    TT-SVD's fidelity need not be monotone at machine precision. We perform
    a small upward scan after a coarse bracketing search; for the smooth
    Heston distributions tested here, the relevant bond is typically small.
    """
    n = amplitudes.size.bit_length() - 1
    max_bond = 1 << (n // 2)

    def score(chi: int) -> float:
        if chi not in cache:
            cache[chi] = float(decompose_state(amplitudes, max_bond=chi).fidelity)
        return cache[chi]

    if score(1) >= fidelity - 1e-12:
        return 1, score(1)
    upper = 2
    while upper < max_bond and score(upper) < fidelity - 1e-12:
        upper = min(2 * upper, max_bond)
    if score(upper) < fidelity - 1e-10:
        raise ArithmeticError(f"TT-SVD did not reach fidelity {fidelity} even at full bond")
    lower = 1
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if score(mid) >= fidelity - 1e-12:
            upper = mid
        else:
            lower = mid
    # Validate candidate and immediately preceding bond. TT-SVD fidelity is
    # expected to increase with bond in practice, but it is not a global
    # optimisation theorem.
    while upper > 1 and score(upper - 1) >= fidelity - 1e-12:
        upper -= 1
    return upper, score(upper)


def sufficient_coarse_fidelity(target: float, lift_fidelity: float) -> float | None:
    """Fidelity needed on the reference MPS to reach target after lifting.

    Uses the Fubini-Study triangle inequality. If the lift alone is too far
    from the fine target, no such sufficient criterion can be given.
    """
    reference_angle = math.acos(math.sqrt(np.clip(lift_fidelity, 0, 1)))
    allowed_angle = math.acos(math.sqrt(target))
    if reference_angle > allowed_angle + 1e-13:
        return None
    return math.cos(max(allowed_angle - reference_angle, 0.0)) ** 2


def benchmark_day(day: str, parameters, args):
    x, pdf, _ = heston_pdf(parameters, n_fft=args.n_fft,
                          halfwidth=args.fft_halfwidth,
                          maturity=args.maturity, drift=args.drift)

    by_ref: dict[int, tuple[np.ndarray, float, float]] = {}

    def reference(m):
        if m not in by_ref:
            edges, _, probabilities, retained = discretize_pdf(
                x, pdf, n_qubits=m, cutoff=args.cutoff)
            overlap = continuous_lift_fidelity(x, pdf, edges, probabilities, retained)
            by_ref[m] = (np.sqrt(probabilities), float(retained), overlap)
        return by_ref[m]

    # Choose the least expensive coarse representation satisfying a stated
    # continuum-limit approximation error, rather than allocating 2**96.
    reference_n = None
    for m in args.reference_qubits:
        _, _, f_lift = reference(m)
        if f_lift >= 1 - args.lift_tolerance:
            reference_n = m
            break
    if reference_n is None:
        # Results remain useful as documented estimates, but a conservative
        # sufficient-chi can be unavailable at the chosen fidelity.
        reference_n = args.reference_qubits[-1]

    out = []
    for n in QUBITS:
        exact = n <= 12
        m = n if exact else reference_n
        amplitudes, retained, cont_fidelity = reference(m)
        cache: dict[int, float] = {}
        row = {
            "date": day,
            "n_qubits": n,
            "n_bins": str(1 << n),
            "cutoff": args.cutoff,
            "retained_mass": retained,
            "method": "exact_discrete_TT_SVD" if exact else "coarse_TT_SVD_plus_uniform_fine_qubits",
            "reference_qubits": m,
            "continuous_lift_fidelity": 1.0 if exact else cont_fidelity,
            "lift_tolerance_met": True if exact else cont_fidelity >= 1 - args.lift_tolerance,
        }
        row["bond_2_fidelity_reference"] = cache.setdefault(
            2, float(decompose_state(amplitudes, max_bond=2).fidelity))
        for target in args.fidelities:
            tag = _tag(target)
            # For high n this is a reference-state estimate, not an exact
            # solution for the high-dimensional Heston amplitude vector.
            chi, achieved = min_ttsvd_bond(amplitudes, target, cache)
            row[f"chi_at_fidelity_{tag}"] = chi
            row[f"reference_fidelity_for_{tag}"] = achieved
            if exact:
                row[f"sufficient_chi_{tag}"] = chi
                row[f"conservative_fidelity_lower_bound_{tag}"] = achieved
            else:
                required = sufficient_coarse_fidelity(target, cont_fidelity)
                if required is None:
                    row[f"sufficient_chi_{tag}"] = ""
                    row[f"conservative_fidelity_lower_bound_{tag}"] = ""
                else:
                    enough_chi, reference_fidelity = min_ttsvd_bond(amplitudes, required, cache)
                    alpha = math.acos(math.sqrt(np.clip(reference_fidelity, 0, 1)))
                    beta = math.acos(math.sqrt(np.clip(cont_fidelity, 0, 1)))
                    row[f"sufficient_chi_{tag}"] = enough_chi
                    row[f"conservative_fidelity_lower_bound_{tag}"] = math.cos(alpha + beta) ** 2
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--n-fft", type=int, default=16384)
    parser.add_argument("--fft-halfwidth", type=float, default=3.5)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--drift", type=float, default=0.05)
    parser.add_argument("--fidelities", nargs="+", type=float, default=TARGET_FIDELITIES)
    parser.add_argument("--reference-qubits", nargs="+", type=int, default=(12, 14, 16),
                        help="Candidate compressed resolutions used for n>=24")
    parser.add_argument("--lift-tolerance", type=float, default=1e-6,
                        help="Maximum target-vs-uniform-lift continuum infidelity")
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()
    if not (0 < args.cutoff < args.fft_halfwidth):
        parser.error("require 0 < cutoff < fft-halfwidth")
    if not (0 < args.lift_tolerance < 1):
        parser.error("lift-tolerance must be between 0 and 1")
    if not all(12 <= q <= 18 for q in args.reference_qubits):
        parser.error("reference-qubits must be between 12 and 18")
    if list(args.reference_qubits) != sorted(set(args.reference_qubits)):
        parser.error("reference-qubits must be sorted and unique")
    if not all(0 < t <= 1 for t in args.fidelities):
        parser.error("all fidelity targets must be in (0,1]")
    if args.max_states is not None and args.max_states <= 0:
        parser.error("max-states must be positive")

    calibrations = read_calibrations(args.input)
    if args.max_states is not None:
        calibrations = calibrations[:args.max_states]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        print(f"Overwriting {args.output}", flush=True)
    # Stream rows: do not hold results for all dates in memory.
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = None
        for i, (day, params) in enumerate(calibrations, 1):
            rows = benchmark_day(day, params, args)
            if writer is None:
                writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            if i % 10 == 0 or i == len(calibrations):
                fp.flush()
                print(f"Processed {i}/{len(calibrations)} dates", flush=True)
    print(f"Saved {len(calibrations)*len(QUBITS)} records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
