"""Compare both tail-based MPS selectors to fixed one-/two-layer workflows.

Uses saved conditional MPS fidelities, without rerunning Heston or MPS.
Fit the two independent conservative thresholds on 2019 ONLY and evaluate
on held-out 2020 dates. Defaults to the stricter F_target=0.99842.

From the repo root:
    python -m cutoff_windows.compare_workflows
    python -m cutoff_windows.compare_workflows --target 0.99845

OVERWRITES existing comparison files in cutoff_windows/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "daily_results.csv"
TARGET = 0.99842
TAIL = "standardized_tail_mass_3sigma"
KURT = "excess_kurtosis"
FEATURE_BY_WORKFLOW = {"tail_selector": TAIL, "kurtosis_selector": KURT}
WORKFLOWS = ("always_one", "always_two", "tail_selector", "kurtosis_selector")
NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK = 2


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = (
        "date", TAIL, KURT, "standardized_retained_mass",
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


def learn_conservative_cutoff(training, target, feature):
    """Choose 1 layer iff feature is greater than every 2019 failure.

    Both classes must occur in training. In particular, at targets >= about
    0.99848 the 2019 sample has no successful 1-layer states; refuse to
    label the resulting degenerate rule a trained selector.
    """
    failures = [
        r[feature] for r in training
        if r["standardized_mps_layer_1_fidelity"] < target
    ]
    successes = [
        r for r in training
        if r["standardized_mps_layer_1_fidelity"] >= target
    ]
    if not failures or not successes:
        raise ValueError(
            f"Training 2019 has {len(successes)} one-layer successes "
            f"and {len(failures)} failures at F={target}. "
            "Both classes are needed for this selector protocol. "
            "Use a lower threshold or train on additional historical data."
        )
    return max(failures)


def choose_layer(row, workflow, cutoffs):
    if workflow == "always_one":
        return 1
    if workflow == "always_two":
        return 2
    if workflow in FEATURE_BY_WORKFLOW:
        feature = FEATURE_BY_WORKFLOW[workflow]
        return 1 if row[feature] > cutoffs[workflow] else 2
    raise ValueError(f"Unknown workflow: {workflow}")


def benchmark(rows, sample, workflow, cutoffs, target, n_qubits):
    if not rows:
        raise ValueError(f"Empty sample: {sample}")
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")
    feature = FEATURE_BY_WORKFLOW.get(workflow)
    cutoff = cutoffs.get(workflow)
    daily = []
    for row in rows:
        layers = choose_layer(row, workflow, cutoffs)
        fidelity = row[f"standardized_mps_layer_{layers}_fidelity"]
        retained = row["standardized_retained_mass"]
        passed = fidelity >= target
        blocks = layers * (n_qubits - 1)
        nominal_cnot = blocks * NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK
        daily.append({
            "date": row["date"], "sample": sample, "workflow": workflow,
            "target_fidelity": target, "feature": feature or "",
            "feature_value": row[feature] if feature else "",
            "train_cutoff": cutoff if cutoff is not None else "",
            "n_qubits": n_qubits, "chosen_layers": layers,
            "conditional_fidelity": fidelity,
            "standardized_retained_mass": retained,
            "full_domain_fidelity_upper_proxy": retained * fidelity,
            "meets_conditional_target": passed,
            "conditional_shortfall": max(0.0, target - fidelity),
            "two_qubit_blocks": blocks, "nominal_cnot_count": nominal_cnot,
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
        "mean_nominal_cnots": float(np.mean(cnot)),
    }
    return daily, totals


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty comparison")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(input_path, output_dir, target=TARGET, n_qubits=8):
    if not 0 < target <= 1:
        raise ValueError("Target fidelity must be between 0 and 1")
    rows = load_rows(input_path)
    training = [r for r in rows if r["date"].startswith("2019")]
    test = [r for r in rows if r["date"].startswith("2020")]
    if not training or not test:
        raise ValueError("Expected both 2019 training and 2020 held-out states")
    cutoffs = {workflow: learn_conservative_cutoff(training, target, feature)
               for workflow, feature in FEATURE_BY_WORKFLOW.items()}
    all_daily = []
    summaries = []
    for sample, group in (("training_2019", training),
                          ("holdout_2020", test),
                          ("all_2019_2020", rows)):
        for workflow in WORKFLOWS:
            daily, summary = benchmark(group, sample, workflow, cutoffs,
                                       target, n_qubits)
            all_daily.extend(daily)
            summaries.append(summary)
    holdout = {r["workflow"]: r for r in summaries
               if r["sample"] == "holdout_2020"}
    one = holdout["always_one"]
    two = holdout["always_two"]
    conclusions = {}
    for name in FEATURE_BY_WORKFLOW:
        selector = holdout[name]
        conclusions[name] = {
            "cnot_saving_vs_two":
                1 - selector["total_nominal_cnots"] / two["total_nominal_cnots"],
            "cnot_increase_vs_one":
                selector["total_nominal_cnots"] / one["total_nominal_cnots"] - 1,
            "fidelity_failure_reduction_vs_one":
                one["states_below_target"] - selector["states_below_target"],
            "selector_all_pass": selector["states_below_target"] == 0,
            "selector_cheaper_than_two":
                selector["total_nominal_cnots"] < two["total_nominal_cnots"],
            "selector_cheaper_than_one":
                selector["total_nominal_cnots"] < one["total_nominal_cnots"],
            "unnecessary_two_layer_dates":
                sum(r["chosen_layers"] == 2 and
                    next(x for x in test if x["date"] == r["date"])[
                        "standardized_mps_layer_1_fidelity"] >= target
                    for r in all_daily
                    if r["sample"] == "holdout_2020" and
                       r["workflow"] == name),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "workflow_comparison_daily.csv", all_daily)
    write_csv(output_dir / "workflow_comparison_summary.csv", summaries)
    summary = {
        "training_period": "2019", "evaluation_period": "2020",
        "target_fidelity": target,
        "selector_features": FEATURE_BY_WORKFLOW,
        "learned_training_cutoffs": cutoffs,
        "n_qubits": n_qubits,
        "nominal_cnot_per_two_qubit_block": NOMINAL_CNOTS_PER_TWO_QUBIT_BLOCK,
        "source_daily_results": str(input_path),
        "comparisons": summaries,
        "holdout_conclusions": conclusions,
        "caveats": [
            "Saved conditional fidelities were reused; no circuits were re-optimised.",
            "CNOT counts are nominal (2 per real two-qubit MPS gate), not compiled hardware counts.",
            "No overhead is charged for computing distributional predictors.",
            "Retained mass times conditional fidelity is a full-domain overlap proxy, not a measured full-domain fidelity.",
            "Both selector cutoffs are trained only on 2019; 2020 is held out.",
            "Results depend on the target fidelity and sampled Heston regime.",
        ],
    }
    (output_dir / "workflow_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"2020 holdout: {len(test)} states, target F={target:.6f}")
    print(f"2019 training: {len(training)} states; one-layer successes: "
          f"{sum(r['standardized_mps_layer_1_fidelity'] >= target for r in training)}")
    for name, cut in cutoffs.items():
        print(f"{name}: one layer if {FEATURE_BY_WORKFLOW[name]} > {cut:.12f}")
    print(f"{'Workflow':<19} {'Pass/total':>11} {'2 layers':>9} "
          f"{'Mean F':>12} {'CNOT total':>11} {'CNOT/state':>11}")
    for name in WORKFLOWS:
        result = holdout[name]
        print(f"{name:<19} {result['states_meeting_target']:>3}/"
              f"{result['n_states']:<7} {result['two_layer_states']:>9} "
              f"{result['mean_conditional_fidelity']:>12.9f} "
              f"{result['total_nominal_cnots']:>11} "
              f"{result['mean_nominal_cnots']:>11.3f}")
    print("Overwrote workflow_comparison_daily.csv, "
          "workflow_comparison_summary.csv and workflow_comparison_summary.json")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("--target", type=float, default=TARGET)
    p.add_argument("--qubits", type=int, default=8)
    args = p.parse_args()
    compare(args.input, args.output_dir, args.target, args.qubits)


if __name__ == "__main__":
    main()
