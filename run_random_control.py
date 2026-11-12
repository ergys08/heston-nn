"""Train all 3 ansatze on one shared 1,000-state first-anchor random control.

Run: python run_random_control.py
     python run_random_control.py --epochs 10 --device cpu
Save per-architecture checkpoints, train history, per-date future fidelities,
random amplitudes, distances and the experiment manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_network.synthetic_random_states import generate_random_states
from neural_network.training_workflow import (
    read_heston_states, train_network, evaluate_network, nearest_fs_distances,
    save_checkpoint,
)

ARCHITECTURES = {"brickwork": 18, "chain": 18, "ring": 16}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-csv", default="heston/heston_params_2013.csv")
    p.add_argument("--first-150", type=int, default=150,
                   help="Training cutoff for test split; only first row anchors control")
    p.add_argument("--num-synthetic", type=int, default=1000)
    p.add_argument("--max-fs-distance", type=float, default=0.10095073648716599)
    p.add_argument("--min-radius-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES),
                   default=list(ARCHITECTURES))
    p.add_argument("--output-dir", default="results/random_first_state_control")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates, heston_states = read_heston_states(args.historical_csv, num_qubits=8)
    if args.first_150 < 1 or len(dates) <= args.first_150:
        p.error("Need at least one anchor and one future state")
    first, last_train_day = heston_states[0], dates[args.first_150 - 1]
    test_dates, test_states = dates[args.first_150:], heston_states[args.first_150:]
    print(f"Anchor: {dates[0]}; training cutoff: {last_train_day}; "
          f"{len(test_dates)} future dates", flush=True)

    random = generate_random_states(
        first, num_states=args.num_synthetic,
        max_fs_distance=args.max_fs_distance,
        min_radius_fraction=args.min_radius_fraction, seed=args.seed,
    )
    np.savez_compressed(out / "random_training_states.npz", states=random.states,
                        first_historical_state=first, fs_distances=random.fs_distances,
                        first_historical_date=dates[0])
    future_hash = hashlib.sha256(np.ascontiguousarray(test_states).tobytes()).hexdigest()
    manifest = {
        "design": "single-anchor naive amplitude-space control",
        "first_historical_anchor": dates[0],
        "training_cutoff": last_train_day, "last_test_date": test_dates[-1],
        "future_test_dates": len(test_dates), "future_state_sha256": future_hash,
        "model_input": "real 256-element amplitude vector",
        "num_synthetic": args.num_synthetic, "training_includes_anchor": False,
        "historical_training_centres": 1,
        "sampler": "iid abs(N(0,1)) direction; spherical interpolation from first anchor",
        "max_fs_distance_rad": random.max_fs_distance,
        "min_radius_fraction": random.min_radius_fraction,
        "observed_min_fs_rad": float(random.fs_distances.min()),
        "observed_mean_fs_rad": float(random.fs_distances.mean()),
        "observed_max_fs_rad": float(random.fs_distances.max()),
        "radius_matched_to_historical_augmented_baseline": True,
        "uses_heston_parameters_for_random_sampling": False,
        "qubits": 8, "rotation": "yz",
        "architectures": {k: ARCHITECTURES[k] for k in args.architectures},
        "mlp": "Linear(256,512)-Tanh-Linear(512,num_parameters)-Tanh",
        "optimizer": "Adam", "loss": "1-|inner product|^2",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "seed": args.seed, "device": args.device,
        "caveat": "Differs in number of historical centres and amplitude smoothness; not a single-factor control.",
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {args.num_synthetic} random states; "
          f"max distance {random.fs_distances.max():.6f} rad", flush=True)

    table = []
    for topology in args.architectures:
        layers = ARCHITECTURES[topology]
        directory = out / topology
        directory.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {topology} ({layers} layers) ===", flush=True)
        progress = directory / "training_progress.pt"
        if args.resume and not progress.exists():
            p.error(f"Missing resume checkpoint: {progress}")
        model, history = train_network(
            random.states, num_qubits=8, num_layers=layers,
            topology=topology, rotation="yz", epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, seed=args.seed, device=args.device,
            progress_path=progress, resume_from=progress if args.resume else None,
        )
        _write_csv(directory / "train_history.csv", history)
        save_checkpoint(model, directory / "model.pt", cutoff_date=last_train_day, seed=args.seed)
        fidelities = evaluate_network(model, test_states, batch_size=args.batch_size, device=args.device)
        distances = nearest_fs_distances(test_states, first[None, :])
        _write_csv(directory / "future_test_results.csv", [
            {"date": d, "fidelity": float(f), "infidelity": float(1 - f),
             "first_anchor_fs_distance_rad": float(fs)}
            for d, f, fs in zip(test_dates, fidelities, distances)
        ])
        result = {
            "ansatz": topology, "layers": layers,
            "num_angles": model.num_parameters,
            "train_fidelity_last_epoch": float(history[-1]["train_fidelity"]),
            "mean_future_fidelity": float(fidelities.mean()),
            "median_future_fidelity": float(np.median(fidelities)),
            "worst_future_fidelity": float(fidelities.min()),
            "future_fidelity_ge_0_99": int(np.count_nonzero(fidelities >= 0.99)),
            "num_future_states": len(fidelities), "future_state_sha256": future_hash,
            "epochs": args.epochs,
        }
        table.append(result)
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Frozen future mean fidelity:", result["mean_future_fidelity"], flush=True)

    _write_csv(out / "ansatz_comparison.csv", table)
    (out / "ansatz_comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    for row in table:
        print(f"{row['ansatz']:10} layers={row['layers']:2} angles={row['num_angles']:3} "
              f"future mean={row['mean_future_fidelity']:.6f} "
              f">=0.99: {row['future_fidelity_ge_0_99']}/{row['num_future_states']}", flush=True)
    print("Results directory:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
