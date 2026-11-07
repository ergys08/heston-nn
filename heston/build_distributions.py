"""Load historical Heston calibrations and build in-memory distributions."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .distributions import HestonParameters, discretize_pdf, heston_pdf

DATA_FILE = Path(__file__).with_name("heston_params_2013.csv")
DATA_FILES = {
    2013: DATA_FILE,
    2020: Path(__file__).with_name("heston_params_2020.csv"),
}


@dataclass
class DistributionBatch:
    dates: list[str]
    parameters: list[HestonParameters]
    x: np.ndarray                 # Full log-return grid.
    pdfs: np.ndarray              # [date, grid position]
    bin_edges: np.ndarray
    bin_masses: np.ndarray        # [date, bin], before conditioning.
    probabilities: np.ndarray    # [date, bin], sum to one.
    retained_masses: np.ndarray   # [date], probability inside cutoff.

    @property
    def amplitudes(self):
        return np.sqrt(self.probabilities)


def read_calibrations(path=DATA_FILE, *, exclude_degenerate=False):
    """Read chronological Heston calibrations from 2013 or 2020.

    Both published datasets are kept intact. The original 2013 'sigma'
    and 2020 'eta' headers denote the same Heston vol-of-vol parameter,
    called 'xi' in HestonParameters. The 2020 'degenerate' flag is
    preserved upstream; it is excluded only when requested.
    """
    path = Path(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        required = {"date", "v0", "kappa", "theta", "rho"}
        if not required.issubset(fields):
            raise ValueError(f"Missing Heston calibration columns in {path}")
        vol_header = "sigma" if "sigma" in fields else "eta" if "eta" in fields else None
        if vol_header is None:
            raise ValueError(f"Expected sigma or eta vol-of-vol column in {path}")
        if exclude_degenerate and "degenerate" not in fields:
            raise ValueError(f"Cannot exclude degenerate rows: no flag in {path}")
        for row in reader:
            if exclude_degenerate and row["degenerate"].strip().lower() in ("true", "1", "yes"):
                continue
            rows.append((row["date"], HestonParameters(
                v0=float(row["v0"]), kappa=float(row["kappa"]),
                theta=float(row["theta"]), xi=float(row[vol_header]), rho=float(row["rho"]),
            )))
    dates = [date for date, _ in rows]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Calibration dates must be unique and chronological")
    return rows


def build_distributions(calibrations, *, n_qubits, cutoff, n_fft, halfwidth,
                        maturity, drift, verify_grid, fft_l1_tolerance):
    """Compute all PDFs and 2**n_qubits bin probabilities; write no files.

    With verify_grid=True, independently repeat each inversion at twice the
    FFT resolution; reject results with bin-mass L1 discrepancy > fft_l1_tolerance.
    """
    if not calibrations:
        raise ValueError("At least one calibration is required")
    if not np.isfinite(fft_l1_tolerance) or fft_l1_tolerance < 0:
        raise ValueError("fft_l1_tolerance must be finite and nonnegative")

    dates, parameters, pdfs, masses, probs, retained = [], [], [], [], [], []
    for date, params in calibrations:
        x, pdf, _ = heston_pdf(params, n_fft=n_fft, halfwidth=halfwidth,
                               maturity=maturity, drift=drift)
        edges, mass, prob, keep = discretize_pdf(x, pdf, n_qubits=n_qubits,
                                                cutoff=cutoff)
        if verify_grid:
            x2, pdf2, _ = heston_pdf(params, n_fft=2 * n_fft,
                                    halfwidth=halfwidth, maturity=maturity, drift=drift)
            _, mass2, _, _ = discretize_pdf(x2, pdf2, n_qubits=n_qubits,
                                            cutoff=cutoff)
            if np.sum(np.abs(mass - mass2)) > fft_l1_tolerance:
                raise ArithmeticError(f"FFT resolution check failed on {date}")
        dates.append(date)
        parameters.append(params)
        pdfs.append(pdf)
        masses.append(mass)
        probs.append(prob)
        retained.append(keep)

    return DistributionBatch(dates, parameters, x, np.stack(pdfs), edges,
                             np.stack(masses), np.stack(probs), np.array(retained))
