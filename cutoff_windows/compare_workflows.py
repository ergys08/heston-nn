"""Benchmark 1-layer, 2-layer and tail-selected MPS preparation.

Uses already computed eight-qubit Heston state fidelities. Does not rerun
the MPS optimiser: all three workflows use the same saved state preparations.
2019 is the training set, and 2020 is a held-out evaluation set.

Run from repository root:
    python -m cutoff_windows.compare_workflows
    python -m cutoff_windows.compare_workflows --target 0.99835
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "daily_results.csv"
TARGET = 0.99835
FEATURE = "standardized_tail_mass_3sigma"
NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK = 2


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = (
        "date", FEATURE, "excess_kurtosis", "standardized_retained_mass",
        "standardized_mps_layer_1_fidelity",
        "standardized_mps_layer_2_fidelity",
    )
    if not rows:
        raise ValueError("No saved daily states")
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    for row in rows:
        for col in required[1:]:
            row[col] = float(row[col])
        if not all(np.isfinite(row[col]) for col in required[1:]):
            raise ValueError(f"Non-finite state metric for {row['date']}")
        if not (0 <= row["standardized_retained_mass"] <= 1):
            raise ValueError("Invalid retained probability")
    return rows


def learn_conservative_cutoff(training, target, feature=FEATURE):
    """Choose 1 layer only above all known 1-layer failures in 2019.

    No evaluation labels influence this cutoff. Requiring a failure in the
    training set avoids claiming a guarantee from an empty negative class.
    """
    failures = [
        r[feature] for r in training
        if r["standardized_mps_layer_1_fidelity"] < target
    ]
    if not failures:
        raise ValueError("No 1-layer failures in training; cannot calibrate cutoff")
    return max(failures)


def choose_layer(row, workflow, cutoff, feature=FEATURE):
    if workflow == "always_one":
        return 1
    if workflow == "always_two":
        return 2
    if workflow == "tail_selector":
        return 1 if row[feature] > cutoff else 2
    raise ValueError(f"Unknown workflow: {workflow}")


def benchmark(rows, sample, workflow, cutoff, target, n_qubits, feature=FEATURE):
    if not rows:
        raise ValueError(f"Empty sample: {sample}")
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")
    daily = []
    for row in rows:
        layers = choose_layer(row, workflow, cutoff, feature)
        fidelity = row[f"standardized_mps_layer_{layers}_fidelity"]
        retained = row["standardized_retained_mass"]
        passed = fidelity >= target
        # Each iterative MPS layer uses (N-1) 2-qubit blocks.
        blocks = layers * (n_qubits - 1)
        nominal_cnot = blocks * NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK
        daily.append({
            "date": row["date"], "sample": sample, "workflow": workflow,
            "target_fidelity": target, "feature": feature,
            "feature_value": row[feature], "train_cutoff": cutoff,
            "n_qubits": n_qubits, "chosen_layers": layers,
            "conditional_fidelity": fidelity,
            "standardized_retained_mass": retained,
            "full_domain_fidelity_upper_proxy": retained * fidelity,
            "meets_conditional_target": passed,
            "conditional_shortfall": max(0.0, target - fidelity),
            "two_qubit_blocks": blocks, "nominal_cnot_count": nominal_cnot
        })
    fidelities = np.array([r["conditional_fidelity"] for r in daily])
    cnot = np.array([r["nominal_cnot_count"] for r in daily])
    shortfalls = np.array([r["conditional_shortfall"] for r in daily])
    layers = np.array([r["chosen_layers"] for r in daily])
    n_pass = sum(r["meets_conditional_target"] for r in daily)
    totals = {
        "sample": sample, "workflow": workflow,
        "target_fidelity": target, "n_states": len(rows),
        "one_layer_states": int(np.sum(layers == 1)),
        "two_layer_states": int(np.sum(layers == 2)),
        "states_meeting_target": n_pass,
        "states_below_target": len(rows) - n_pass,
        "pass_rate": n_pass / len(rows),
        "mean_conditional_fidelity": float(np.mean(fidelities)),
        "min_conditional_fidelity": float(np.min(fidelities)),
        "mean_conditional_shortfall": float(np.mean(shortfalls)),
        "worst_conditional_shortfall": float(np.max(shortfalls)),
        "mean_retained_mass": float(np.mean(
            [r["standardized_retained_mass"] for r in daily])),
        "mean_full_domain_fidelity_upper_proxy": float(np.mean(
            [r["full_domain_fidelity_upper_proxy"] for r in daily])),
        "total_two_qubit_blocks": int(sum(r["two_qubit_blocks"] for r in daily)),
        "total_nominal_cnots": int(np.sum(cnot)),
        "mean_nominal_cnots": float(np.mean(cnot))
    }
    return daily, totals


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty comparison")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(input_path, output_dir, target=TARGET, n_qubits=8,
            feature=FEATURE):
    if not 0 < target <= 1:
        raise ValueError("Target fidelity must be between 0 and 1")
    rows = load_rows(input_path)
    training = [r for r in rows if r["date"].startswith("2019")]
    test = [r for r in rows if r["date"].startswith("2020")]
    if not training or not test:
        raise ValueError("Expected both 2019 training and 2020 held-out states")
    cutoff = learn_conservative_cutoff(training, target, feature)
    all_daily = []
    summaries = []
    for sample, group in (("training_2019", training),
                          ("holdout_2020", test),
                          ("all_2019_2020", rows)):
        for workflow in ("always_one", "always_two", "tail_selector"):
            d, s = benchmark(group, sample, workflow, cutoff, target,
                             n_qubits, feature)
            all_daily.extend(d)
            summaries.append(s)
    holdout = {r["workflow"]: r for r in summaries
               if r["sample"] == "holdout_2020"}
    selector = holdout["tail_selector"]
    one = holdout["always_one"]
    two = holdout["always_two"]
    if not (one["states_below_target"] or selector["states_below_target"] == 0):
        raise AssertionError("Unexpected comparison state")
    conclusions = {
        "cnot_saving_vs_two": 1 - selector["total_nominal_cnots"] /
                              two["total_nominal_cnots"],
        "cnot_increase_vs_one": selector["total_nominal_cnots"] /
                                 one["total_nominal_cnots"] - 1,
        "fidelity_failure_reduction_vs_one":
            one["states_below_target"] - selector["states_below_target"],
        "selector_all_pass": selector["states_below_target"] == 0,
        "selector_cheaper_than_two": selector["total_nominal_cnots"] <
                                     two["total_nominal_cnots"],
        "selector_cheaper_than_one": selector["total_nominal_cnots"] <
                                     one["total_nominal_cnots"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "workflow_comparison_daily.csv", all_daily)
    write_csv(output_dir / "workflow_comparison_summary.csv", summaries)
    summary = {
        "training_period": "2019", "evaluation_period": "2020",
        "target_fidelity": target, "feature": feature,
        "learned_training_cutoff": cutoff, "n_qubits": n_qubits,
        "nominal_cnot_per_two_qubit_block": NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK,
        "source_daily_results": str(input_path), "comparisons": summaries,
        "holdout_conclusions": conclusions,
        "caveats": [
            "Saved conditional fidelities were reused; no circuits were re-optimised.",
            "CNOT counts are nominal (2 per real two-qubit MPS gate), not compiled hardware counts.",
            "No overhead is charged for evaluating the distributional feature.",
            "Retained mass times conditional fidelity is an idealised full-domain overlap proxy, not a measured full-domain circuit fidelity.",
            "The selector is trained only on 2019; 2020 is held out.",
            "The selector optimises fidelity compliance and gate usage, not maximum attainable fidelity."
        ]
    }
    (output_dir / "workflow_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("2020 holdout (target F = {:.5f})".format(target))
    print("Rule: select 1 layer if {} > {:.12f}; else 2".format(feature, cutoff))
    print("{:<17} {:>10} {:>10} {:>14} {:>15}".format(
        "Workflow", "Passes", "CNOTs", "Mean fidelity", "Mean CNOTs"))
    for name in ("always_one", "always_two", "tail_selector"):
        result = holdout[name]
        print("{:<17} {:>4}/{:<5} {:>10} {:>14.9f} {:>15.3f}".format(
            name, result["states_meeting_target"], result["n_states"],
            result["total_nominal_cnots"], result["mean_conditional_fidelity"],
            result["mean_nominal_cnots"]))
    print("Saved comparisons in", output_dir.resolve())
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("--target", type=float, default=TARGET)
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--feature", choices=(FEATURE, "excess_kurtosis"),
                   default=FEATURE)
    args = p.parse_args()
    compare(args.input, args.output_dir, args.target, args.qubits, args.feature)


if __name__ == "__main__":
    main()
