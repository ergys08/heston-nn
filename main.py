"""Run the first 2013 Heston SuperEncoder train-then-test experiment.

Usage from the heston-nn repository root:
    python main.py
    python main.py --epochs 10 --device cuda
    python main.py --future-csv path/to/later_calibrations.csv

The FIRST 150 dated calibrations are training anchors, with 850 synthetic
amplitude-space states. Subsequent dates are evaluated ONCE with a frozen MLP.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neural_network.synthetic_states import generate_synthetic_states
from neural_network.training_workflow import (
    evaluate_network, nearest_fs_distances, read_heston_states,
    save_checkpoint, seed_everything, train_network,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-csv", default="heston/heston_params_2013.csv")
    parser.add_argument("--future-csv", default=None,
                        help="Optional separate CSV of later Heston calibrations for frozen evaluation")
    parser.add_argument("--train-count", type=int, default=150)
    parser.add_argument("--num-synthetic", type=int, default=850)
    parser.add_argument("--radius-quantile", type=float, default=0.75)
    parser.add_argument("--max-fs-distance", type=float, default=None,
                        help="Override local perturbation radius in Fubini--Study radians")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--topology", choices=["brickwork", "chain", "ring"], default="brickwork")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--rotation", choices=["yz", "rot"], default="yz")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (MPS lacks complex128 support)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/first_heston_network")
    parser.add_argument("--resume", action="store_true", help="Resume from saved per-epoch training progress")
    args = parser.parse_args()
    if args.train_count < 2:
        parser.error("--train-count must be at least 2")
    if args.layers is None:
        args.layers = {"brickwork": 18, "chain": 18, "ring": 16}[args.topology]

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Reading historical calibrations...", flush=True)
    all_dates, all_states = read_heston_states(args.historical_csv, num_qubits=8)
    if len(all_dates) <= args.train_count and args.future_csv is None:
        parser.error("Historical CSV must contain dates after the training cutoff")
    train_dates = all_dates[:args.train_count]
    anchors = all_states[:args.train_count]
    cutoff_date = train_dates[-1]
    if args.future_csv is None:
        future_dates = all_dates[args.train_count:]
        future_states = all_states[args.train_count:]
    else:
        future_dates, future_states = read_heston_states(args.future_csv, num_qubits=8)
        if future_dates[0] <= cutoff_date:
            parser.error("Future dates must be strictly after the training cutoff date")

    print(f"Training anchors: {len(anchors)} ({train_dates[0]} through {cutoff_date})")
    print(f"Held-out future states: {len(future_states)} ({future_dates[0]} through {future_dates[-1]})")
    augmented = generate_synthetic_states(
        anchors, args.num_synthetic, seed=args.seed,
        max_fs_distance=args.max_fs_distance, radius_quantile=args.radius_quantile,
    )
    training_states = np.concatenate([anchors, augmented.states], axis=0)
    print(f"Generated {len(augmented.states)} synthetic states; "
          f"max FS radius={augmented.max_fs_distance:.6f} rad; "
          f"actual max={augmented.fs_distances.max(initial=0):.6f} rad", flush=True)
    # Exact first-network settings; recorded before fitting or future testing.
    settings = {
        "experiment": "150 historical + 850 amplitude-space augmented states",
        "historical_csv": args.historical_csv,
        "future_csv": args.future_csv,
        "train_start_date": train_dates[0],
        "training_cutoff": cutoff_date,
        "future_start_date": future_dates[0],
        "historical_training_states": len(anchors),
        "synthetic_training_states": len(augmented.states),
        "augmentation": {
            "method": "smooth Gaussian noise in amplitude space; positive multiplicative reference, then spherical interpolation",
            "parameters_perturbed": False,
            "parent_selection": "balanced and randomly shuffled",
            "max_fs_radius_rad": augmented.max_fs_distance,
            "radius_quantile": args.radius_quantile,
            "min_radius_fraction": 0.1,
            "max_radius_fraction": 1.0,
            "noise_gaussian_smoothing_sigma_bins": 2.5,
            "noise_kernel_halfwidth_bins": 8,
            "reference_strength_uniform": [1.1, 2.1],
            "exact_per_sample_perturbations": "training_states.npz:synthetic_fs_distances",
            "exact_parent_indices": "training_states.npz:synthetic_parent_indices",
        },
        "training": {
            "architecture": args.topology,
            "qubits": 8,
            "layers": args.layers,
            "rotation": args.rotation,
            "network": "Linear(256,512)-Tanh-Linear(512,P)-Tanh",
            "loss": "1-|overlap|^2",
            "optimizer": "Adam",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
        },
        "test": {"num_future_states": len(future_states), "frozen_network": True},
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "training_states.npz", historical=anchors,
        synthetic=augmented.states, synthetic_parent_indices=augmented.parent_indices,
        synthetic_fs_distances=augmented.fs_distances,
        historical_dates=np.asarray(train_dates),
    )

    progress = output_dir / "training_progress.pt"
    if args.resume and not progress.exists():
        parser.error(f"Missing checkpoint for --resume: {progress}")
    model, history = train_network(
        training_states, num_qubits=8, num_layers=args.layers,
        topology=args.topology, rotation=args.rotation, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device,
        progress_path=progress, resume_from=progress if args.resume else None,
    )
    write_csv(output_dir / "train_history.csv", history)
    save_checkpoint(model, output_dir / "model.pt", cutoff_date=cutoff_date, seed=args.seed)

    # Future states are not passed to augmentation or training and the net is frozen.
    test_fidelities = evaluate_network(
        model, future_states, batch_size=args.batch_size, device=args.device,
    )
    distances = nearest_fs_distances(future_states, anchors)
    rows = [
        {"date": day, "fidelity": float(fid), "infidelity": float(1 - fid),
         "nearest_historical_fs_rad": float(dist)}
        for day, fid, dist in zip(future_dates, test_fidelities, distances)
    ]
    write_csv(output_dir / "future_test_results.csv", rows)
    summary = {
        "training_cutoff": cutoff_date,
        "historical_training_states": len(anchors),
        "synthetic_training_states": len(augmented.states),
        "future_test_states": len(future_states),
        "topology": args.topology,
        "layers": args.layers,
        "rotation": args.rotation,
        "augmentation_max_fs_distance_rad": augmented.max_fs_distance,
        "mean_future_fidelity": float(test_fidelities.mean()),
        "median_future_fidelity": float(np.median(test_fidelities)),
        "min_future_fidelity": float(test_fidelities.min()),
        "fraction_future_fidelity_above_0_99": float(np.mean(test_fidelities >= .99)),
        "last_train_epoch_fidelity": history[-1]["train_fidelity"],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nFrozen-network future test:")
    print(f"  mean fidelity:   {summary['mean_future_fidelity']:.6f}")
    print(f"  median fidelity: {summary['median_future_fidelity']:.6f}")
    print(f"  worst fidelity:  {summary['min_future_fidelity']:.6f}")
    print(f"  fraction >= .99: {summary['fraction_future_fidelity_above_0_99']:.1%}")
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
