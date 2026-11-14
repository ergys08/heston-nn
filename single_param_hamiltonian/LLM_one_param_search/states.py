"""Build and cache the historical Heston eight-qubit target states.

Encoding is identical to single_param_hamiltonian/ and MPS/: Fourier-inverted
terminal log-return PDF (maturity 1.0, drift 0.05), 256 bins on [-2.5, 2.5],
amplitudes sqrt(conditional bin probability), paper qubit 1 = MSB.

2013: 250 dates, n_fft 16384, halfwidth 3.5 (repository defaults).
2020: 71 dates, n_fft 131072, halfwidth 25 (heston/README.md recommendation).

Subsets are chosen by farthest-point sampling in Fubini-Study distance so that
training sets contain dissimilar states, not just neighbouring dates.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from heston.distributions import HestonParameters, discretize_pdf, heston_pdf  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

ENCODINGS = {
    2013: dict(csv=ROOT / "heston" / "heston_params_2013.csv", xi_column="sigma",
               n_fft=16384, halfwidth=3.5),
    2020: dict(csv=ROOT / "heston" / "heston_params_2020.csv", xi_column="eta",
               n_fft=131072, halfwidth=25.0),
}


def _build(year: int):
    cfg = ENCODINGS[year]
    dates, states, retained, skipped = [], [], [], []
    with Path(cfg["csv"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                p = HestonParameters(
                    v0=float(row["v0"]), kappa=float(row["kappa"]),
                    theta=float(row["theta"]), xi=float(row[cfg["xi_column"]]),
                    rho=float(row["rho"]),
                )
                x, pdf, _ = heston_pdf(p, n_fft=cfg["n_fft"], halfwidth=cfg["halfwidth"],
                                       maturity=1.0, drift=0.05)
                _, _, prob, mass = discretize_pdf(x, pdf, n_qubits=8, cutoff=2.5)
            except (ValueError, ArithmeticError) as err:
                skipped.append((row["date"], str(err)))
                continue
            psi = np.sqrt(prob)
            assert abs(np.linalg.norm(psi) - 1) < 1e-12
            dates.append(row["date"]); states.append(psi); retained.append(float(mass))
    return np.asarray(dates), np.asarray(states), np.asarray(retained), skipped


def load_states(year: int = 2013):
    """Return (dates, states[J,256], retained_mass). Cached in results/."""
    cache = RESULTS / f"states_{year}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return d["dates"], d["states"], d["retained"]
    dates, states, retained, skipped = _build(year)
    np.savez_compressed(cache, dates=dates, states=states, retained=retained)
    if skipped:
        (RESULTS / f"states_{year}_skipped.txt").write_text(
            "\n".join(f"{d}: {e}" for d, e in skipped) + "\n")
    return dates, states, retained


def fs_distance_matrix(states: np.ndarray) -> np.ndarray:
    overlap = np.clip(states @ states.T, -1, 1)
    return np.arccos(overlap)


def farthest_point_subset(states: np.ndarray, n: int, *, start: int = 0) -> list[int]:
    """Greedy farthest-point sampling in FS distance (deterministic)."""
    D = fs_distance_matrix(states)
    chosen = [start]
    mind = D[start].copy()
    while len(chosen) < n:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, D[nxt])
    return chosen


def nested_subsets(states: np.ndarray, sizes=(8, 24, 60)) -> dict[int, list[int]]:
    """Nested FPS subsets; the largest is computed once and prefixes are used."""
    full = farthest_point_subset(states, max(sizes))
    return {n: full[:n] for n in sizes}


if __name__ == "__main__":
    import json
    for year in (2013, 2020):
        dates, states, retained = load_states(year)
        D = fs_distance_matrix(states)
        print(f"{year}: {len(dates)} states, retained mass [{retained.min():.6f},{retained.max():.6f}],"
              f" max FS distance {D.max():.4f} rad (min fidelity {np.cos(D.max())**2:.4f}),"
              f" mean consecutive F {np.mean(np.cos(np.diag(D,1))**2):.5f}")
    dates, states, _ = load_states(2013)
    subs = nested_subsets(states)
    # product-state and mean-state reference numbers
    mean = states.mean(0); mean /= np.linalg.norm(mean)
    F_mean = (states @ mean) ** 2
    print(f"best-single-fixed-state (mean state) fidelity over 2013: mean {F_mean.mean():.5f} min {F_mean.min():.5f}")
    info = {str(n): {"indices": idx, "dates": [str(dates[i]) for i in idx]} for n, idx in subs.items()}
    (RESULTS / "subsets_2013.json").write_text(json.dumps(info, indent=1))
    for n, idx in subs.items():
        print(n, [str(dates[i]) for i in idx][:12], "...")
