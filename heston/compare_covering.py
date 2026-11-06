"""Compare covering counts for 2013 Heston states and matched random walks.

Run from the repository root:

    python -m heston.compare_covering
    python -m heston.compare_covering --n-walks 100 --subspace-dimension 6
    python -m heston.compare_covering --method exact --n-walks 1 --time-limit 120

Default method is a FAST GREEDY SAMPLE-CENTRED COVER, not the exact minimum.
Use --method exact for a certified minimum among candidate centres
(observed states themselves); this may take much longer.

For every random walk, the first vector is precisely the first Heston state,
and all 249 daily Fubini--Study step lengths match in chronological order.
Random walks are real pure quantum states but typically not Heston PDFs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .build_distributions import build_distributions, read_calibrations
from .covering_number import covering_from_fidelities, fidelity_matrix
from .random_walk import consecutive_step_lengths, matched_step_random_walk


DEFAULT_FIDELITIES = (0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9999)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    *,
    n_walks: int = 100,
    seed: int = 12345,
    subspace_dimension: int = 256,
    fidelities: tuple[float, ...] = DEFAULT_FIDELITIES,
    method: str = "greedy",
    time_limit: float | None = None,
    output_dir: Path | None = Path("covering_results"),
    verify_fft: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Return (summary_rows, per_walk_rows); optionally save CSVs."""
    if n_walks < 1:
        raise ValueError("n_walks must be >= 1")
    if not 2 <= subspace_dimension <= 256:
        raise ValueError("subspace_dimension must be in [2, 256] for 8 qubits")
    if not fidelities:
        raise ValueError("At least one fidelity threshold is required")

    data = build_distributions(
        read_calibrations(),
        n_qubits=8,
        cutoff=2.5,
        n_fft=16384,
        halfwidth=3.5,
        maturity=1.0,
        drift=0.05,
        verify_grid=verify_fft,
        fft_l1_tolerance=1e-4,
    )
    historical = data.amplitudes
    steps = consecutive_step_lengths(historical)
    print(
        f"Loaded {len(historical)} Heston states; {historical.shape[1]} amplitudes; "
        f"{len(steps)} chronological steps."
    )
    print(
        f"Walk subspace dimension={subspace_dimension} (sphere dimension "
        f"{subspace_dimension - 1}); {n_walks} seeded walks."
    )
    print(f"Cover algorithm: {method} (candidate centres = observed states)")

    thresholds = tuple(float(f) for f in fidelities)
    historical_fidelities = fidelity_matrix(historical)
    heston_results = [
        covering_from_fidelities(
            historical_fidelities, min_fidelity=f, method=method, time_limit=time_limit
        )
        for f in thresholds
    ]
    random_counts = np.empty((n_walks, len(thresholds)), dtype=np.int32)
    per_walk_rows: list[dict] = []
    for trial in range(n_walks):
        trial_seed = seed + trial
        walk = matched_step_random_walk(
            historical, seed=trial_seed, subspace_dimension=subspace_dimension
        )
        if not np.allclose(walk[0], historical[0], atol=1e-12):
            raise ArithmeticError("Random walk did not start at the first Heston state")
        if not np.allclose(consecutive_step_lengths(walk), steps, atol=1e-7, rtol=0):
            raise ArithmeticError("Random walk did not reproduce the daily step lengths")
        walk_fidelities = fidelity_matrix(walk)
        for col, f in enumerate(thresholds):
            count = covering_from_fidelities(
                walk_fidelities, min_fidelity=f, method=method, time_limit=time_limit
            ).count
            random_counts[trial, col] = count
            per_walk_rows.append(
                {
                    "walk_index": trial,
                    "seed": trial_seed,
                    "subspace_dimension": subspace_dimension,
                    "min_fidelity": f,
                    "radius_rad": float(np.arccos(np.sqrt(f))),
                    "cover_method": method,
                    "random_cover_count": int(count),
                }
            )

    summary_rows: list[dict] = []
    print("\n Fidelity    Radius (rad)   Heston    Random mean    Random 2.5%-97.5%")
    for col, f in enumerate(thresholds):
        vals = random_counts[:, col]
        row = {
            "min_fidelity": f,
            "radius_rad": float(np.arccos(np.sqrt(f))),
            "cover_method": method,
            "proven_sample_centred_optimum": method == "exact",
            "heston_cover_count": heston_results[col].count,
            "random_cover_mean": float(np.mean(vals)),
            "random_cover_std": float(np.std(vals)),
            "random_cover_p025": float(np.quantile(vals, 0.025)),
            "random_cover_p975": float(np.quantile(vals, 0.975)),
            "fraction_random_le_heston": float(np.mean(vals <= heston_results[col].count)),
            "n_walks": n_walks,
            "subspace_dimension": subspace_dimension,
        }
        summary_rows.append(row)
        print(
            f" {f:9.6f}  {row['radius_rad']:12.6f}  "
            f"{row['heston_cover_count']:7d}  "
            f"{row['random_cover_mean']:13.2f}  "
            f"[{row['random_cover_p025']:.1f}, {row['random_cover_p975']:.1f}]"
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "covering_summary.csv", summary_rows)
        _write_csv(output_dir / "covering_random_walks.csv", per_walk_rows)
        print(f"\nResults written to {output_dir.resolve()}")
    return summary_rows, per_walk_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--subspace-dimension", type=int, default=256,
        help="Ambient linear subspace (256 = full real 8-qubit space; 6 = 5D sphere).",
    )
    parser.add_argument("--fidelities", type=float, nargs="+", default=DEFAULT_FIDELITIES)
    parser.add_argument("--method", choices=("greedy", "exact"), default="greedy")
    parser.add_argument("--time-limit", type=float, default=None, help="MILP seconds per solve")
    parser.add_argument("--output-dir", type=Path, default=Path("covering_results"))
    parser.add_argument("--no-verify-fft", action="store_true")
    args = parser.parse_args()
    run_comparison(
        n_walks=args.n_walks,
        seed=args.seed,
        subspace_dimension=args.subspace_dimension,
        fidelities=tuple(args.fidelities),
        method=args.method,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        verify_fft=not args.no_verify_fft,
    )


if __name__ == "__main__":
    main()
