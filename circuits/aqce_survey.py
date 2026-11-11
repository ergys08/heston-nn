"""AQCE topology survey over the historical Heston state trajectory.

Run from the repository root:
    python -m circuits.aqce_survey run --year 2013 --blocks 15 --sweeps 2
    python -m circuits.aqce_survey analyse --year 2013 --min-fidelity .995

Requires the original MNISQ AQCE implementation in
heston/third_party/aqce/AQCE_from_python.py (separately attributed upstream).
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from heston.build_distributions import DATA_FILES, build_distributions, read_calibrations


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare_states(args):
    cache = args.output_dir / "target_states.npz"
    settings = dict(year=args.year, exclude_degenerate=args.exclude_degenerate,
                    n_fft=args.n_fft, halfwidth=args.halfwidth,
                    verify_fft=not args.no_verify_fft)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as loaded:
            recorded = json.loads(str(loaded["settings"].item()))
            if recorded != settings:
                raise ValueError("Cached Heston state settings differ. Use a new --output-dir.")
            return loaded["dates"].tolist(), loaded["amplitudes"].copy()
    calibrations = read_calibrations(
        DATA_FILES[args.year], exclude_degenerate=args.exclude_degenerate
    )
    batch = build_distributions(
        calibrations, n_qubits=8, cutoff=2.5, n_fft=args.n_fft,
        halfwidth=args.halfwidth, maturity=1.0, drift=0.05,
        verify_grid=not args.no_verify_fft, fft_l1_tolerance=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, dates=np.asarray(batch.dates),
                        amplitudes=batch.amplitudes,
                        retained_masses=batch.retained_masses,
                        settings=json.dumps(settings))
    return batch.dates, batch.amplitudes


def run(args):
    try:
        from qulacs import QuantumState
        from heston.third_party.aqce.AQCE_from_python import AQCE_python
    except ImportError as exc:
        raise SystemExit(
            "AQCE requires numpy, qulacs and the upstream AQCE_from_python.py "
            "at heston/third_party/aqce/. See the experiment instructions."
        ) from exc
    dates, amplitudes = prepare_states(args)
    last = len(dates) if args.stop is None else min(args.stop, len(dates))
    if not 0 <= args.start < last:
        raise ValueError("Require 0 <= --start < --stop <= number of dates")
    for i in range(args.start, last):
        record_file = args.output_dir / "states" / f"{i:03d}.json"
        matrix_file = args.output_dir / "unitaries" / f"{i:03d}.npz"
        if record_file.exists() and matrix_file.exists():
            existing = json.loads(record_file.read_text(encoding="utf-8"))
            if (existing["blocks"], existing["sweeps"]) != (args.blocks, args.sweeps):
                raise ValueError("Conflicting AQCE settings in output directory. Use another --output-dir.")
            print(f"[{i+1}/{len(dates)}] {dates[i]}: already completed", flush=True)
            continue
        t0 = time.perf_counter()
        target = QuantumState(8)
        target.load(np.asarray(amplitudes[i], dtype=complex))
        gates = AQCE_python(target, M_0=0, M_delta=1,
                            M_max=args.blocks, N=args.sweeps)
        prepared = QuantumState(8)
        for gate in gates:
            gate.update_quantum_state(prepared)
        fidelity = float(abs(np.vdot(amplitudes[i], prepared.get_vector())) ** 2)
        pairs = [list(map(int, gate.get_target_index_list())) for gate in gates]
        if not all(len(pair) == 2 for pair in pairs):
            raise ValueError("Unexpected non-two-qubit AQCE gate")
        if not np.isfinite(fidelity) or not 0 <= fidelity <= 1 + 1e-8:
            raise ArithmeticError(f"Invalid fidelity for {dates[i]}: {fidelity}")
        matrices = np.stack([np.asarray(gate.get_matrix()) for gate in gates])
        matrix_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(matrix_file, matrices=matrices,
                            pairs=np.asarray(pairs, dtype=int))
        save_json(record_file, dict(index=i, date=dates[i], blocks=args.blocks,
                                    sweeps=args.sweeps, fidelity=fidelity,
                                    elapsed_seconds=round(time.perf_counter()-t0, 3),
                                    pairs=pairs))
        print(f"[{i+1}/{len(dates)}] {dates[i]}: F={fidelity:.8f}; "
              f"blocks={len(pairs)}; {time.perf_counter()-t0:.1f}s", flush=True)
    analyse(args)


def canonical_pairs(record):
    return [tuple(sorted(map(int, pair))) for pair in record["pairs"]]


def multiset_jaccard(a, b):
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 1.0


def ordered_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            previous, dp[j] = dp[j], (
                previous + 1 if x == y else max(dp[j], dp[j-1])
            )
    return dp[-1] / max(len(a), len(b))


def analyse(args):
    files = sorted((args.output_dir / "states").glob("*.json"))
    if not files:
        raise SystemExit(f"No AQCE records found under {args.output_dir / 'states'}")
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    records.sort(key=lambda r: r["index"])
    write_csv(args.output_dir / "per_date.csv", [dict(r, topology=json.dumps(r["pairs"]))
               for r in records],
              ["index", "date", "fidelity", "blocks", "sweeps",
               "elapsed_seconds", "topology"])
    valid = [r for r in records if r["fidelity"] >= args.min_fidelity]
    summary = dict(year=args.year, total_completed=len(records),
                   successful_at_min_fidelity=len(valid),
                   min_fidelity=args.min_fidelity,
                   fraction_passing=len(valid)/len(records),
                   min_observed_fidelity=min(r["fidelity"] for r in records),
                   median_observed_fidelity=float(np.median([r["fidelity"] for r in records])))
    print(f"Completed: {len(records)}; fidelity >= {args.min_fidelity}: {len(valid)}")
    if len(valid) < 2:
        save_json(args.output_dir / "summary.json", summary)
        print("Need at least two sufficiently accurate circuits for topology analysis.")
        return

    sequences = [canonical_pairs(r) for r in valid]
    m = len(valid)
    freq = Counter(edge for seq in sequences for edge in seq)
    state_freq = Counter(edge for seq in sequences for edge in set(seq))
    edge_rows = [dict(edge=f"{edge[0]}-{edge[1]}", occurrences=count,
                      state_count=state_freq[edge], state_fraction=state_freq[edge]/m)
                 for edge, count in freq.most_common()]
    write_csv(args.output_dir / "edge_frequency.csv", edge_rows,
              ["edge", "occurrences", "state_count", "state_fraction"])

    motif_rows = []
    for size in (2, 3, 4):
        supports = Counter()
        for seq in sequences:
            supports.update(set(tuple(seq[k:k+size])
                                for k in range(len(seq)-size+1)))
        motif_rows += [dict(length=size, motif=json.dumps(motif),
                            state_count=count, state_fraction=count/m)
                       for motif, count in supports.most_common()]
    write_csv(args.output_dir / "ordered_motifs.csv", motif_rows,
              ["length", "motif", "state_count", "state_fraction"])

    exact = Counter(tuple(seq) for seq in sequences)
    write_csv(args.output_dir / "exact_topologies.csv",
              [dict(topology=json.dumps(seq), state_count=count)
               for seq, count in exact.most_common()], ["topology", "state_count"])

    seqsim = np.eye(m)
    edge_sim = np.eye(m)
    pair_rows = []
    vectors = None
    cache = args.output_dir / "target_states.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            vectors = d["amplitudes"].copy()
    for i in range(m):
        for j in range(i+1, m):
            order = ordered_similarity(sequences[i], sequences[j])
            overlap = multiset_jaccard(sequences[i], sequences[j])
            seqsim[i, j] = seqsim[j, i] = order
            edge_sim[i, j] = edge_sim[j, i] = overlap
            row = dict(index_i=valid[i]["index"], index_j=valid[j]["index"],
                       date_i=valid[i]["date"], date_j=valid[j]["date"],
                       trading_day_gap=valid[j]["index"]-valid[i]["index"],
                       ordered_similarity=order, multiset_jaccard=overlap)
            if vectors is not None:
                x, y = vectors[valid[i]["index"]], vectors[valid[j]["index"]]
                row["state_squared_fidelity"] = float(abs(np.vdot(x, y))**2)
            pair_rows.append(row)
    columns = ["index_i", "index_j", "date_i", "date_j", "trading_day_gap",
               "ordered_similarity", "multiset_jaccard"]
    if vectors is not None:
        columns.append("state_squared_fidelity")
    write_csv(args.output_dir / "topology_pairs.csv", pair_rows, columns)

    adj = [(i, i+1) for i in range(m-1)
           if valid[i+1]["index"]-valid[i]["index"] == 1]
    adj_values = [seqsim[i, j] for i, j in adj]
    global_mean = float(np.mean(seqsim[np.triu_indices(m, 1)]))
    summary.update(mean_global_ordered_similarity=global_mean,
                   mean_global_edge_overlap=float(np.mean(edge_sim[np.triu_indices(m, 1)])),
                   mean_adjacent_ordered_similarity=(float(np.mean(adj_values))
                                                     if adj_values else None),
                   number_adjacent_pairs=len(adj),
                   number_exact_topologies=len(exact))
    if adj:
        rng = np.random.default_rng(2026)
        observed = float(np.mean(adj_values))
        shuffled = []
        for _ in range(500):
            perm = rng.permutation(m)
            shuffled.append(float(np.mean([seqsim[perm[i], perm[j]] for i, j in adj])))
        summary["chronological_locality_permutation_p"] = (
            1+sum(v >= observed for v in shuffled))/(len(shuffled)+1)
        summary["shuffled_adjacent_mean"] = float(np.mean(shuffled))

    lags = sorted(set(row["trading_day_gap"] for row in pair_rows))
    by_lag = [dict(trading_day_gap=lag,
                   n_pairs=sum(r["trading_day_gap"] == lag for r in pair_rows),
                   mean_ordered_similarity=float(np.mean([
                       r["ordered_similarity"] for r in pair_rows
                       if r["trading_day_gap"] == lag])),
                   mean_edge_overlap=float(np.mean([
                       r["multiset_jaccard"] for r in pair_rows
                       if r["trading_day_gap"] == lag]))) for lag in lags]
    write_csv(args.output_dir / "similarity_by_lag.csv", by_lag,
              ["trading_day_gap", "n_pairs", "mean_ordered_similarity", "mean_edge_overlap"])
    far = sorted((r for r in pair_rows if r["trading_day_gap"] >= 30),
                 key=lambda r: (r["ordered_similarity"], r["multiset_jaccard"]),
                 reverse=True)
    write_csv(args.output_dir / "distant_overlaps.csv", far[:50], columns)

    # Greedy *structural* covering; this is not a state-preparation guarantee.
    uncovered = set(range(m))
    structural = []
    while uncovered:
        centre = max(range(m), key=lambda k: sum(
            edge_sim[k, j] >= args.structural_threshold for j in uncovered))
        covered = {j for j in uncovered if edge_sim[centre, j] >= args.structural_threshold}
        structural.append(dict(reference_date=valid[centre]["date"],
                               reference_index=valid[centre]["index"],
                               newly_covered=len(covered),
                               covered_dates=json.dumps([valid[j]["date"] for j in sorted(covered)])))
        uncovered -= covered
    write_csv(args.output_dir / "structural_representatives.csv", structural,
              ["reference_date", "reference_index", "newly_covered", "covered_dates"])
    summary["structural_representatives"] = len(structural)
    summary["structural_overlap_threshold"] = args.structural_threshold

    windows = []
    final_index = max(r["index"] for r in records) + 1
    for start in range(0, final_index, args.stride):
        subset = [(r, seq) for r, seq in zip(valid, sequences)
                  if start <= r["index"] < start+args.window]
        if len(subset) < 2:
            continue
        local = Counter(edge for _, seq in subset for edge in set(seq))
        for edge, count in local.most_common(10):
            windows.append(dict(start_index=start, end_index=start+args.window-1,
                                n_successful=len(subset), edge=f"{edge[0]}-{edge[1]}",
                                state_fraction=count/len(subset)))
    write_csv(args.output_dir / "window_edge_frequency.csv", windows,
              ["start_index", "end_index", "n_successful", "edge", "state_fraction"])
    save_json(args.output_dir / "summary.json", summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for PNG figures; all CSV and JSON files are saved.")
        return

    plt.figure(figsize=(10, 3.6))
    plt.plot([r["index"] for r in records], [r["fidelity"] for r in records], marker=".", markersize=2)
    plt.axhline(args.min_fidelity, ls="--", color="gray")
    plt.xlabel("2013/2020 calibration index (chronological)")
    plt.ylabel("AQCE squared fidelity")
    plt.tight_layout()
    plt.savefig(args.output_dir / "fidelity_by_date.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(seqsim, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Normalized longest common subsequence")
    plt.xlabel("Successful historical circuit (chronological order)")
    plt.ylabel("Successful historical circuit (chronological order)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "ordered_topology_similarity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(edge_sim, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Multiset Jaccard similarity of qubit pairs")
    plt.xlabel("Successful historical circuit (chronological order)")
    plt.ylabel("Successful historical circuit (chronological order)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "edge_overlap_similarity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    most_common = edge_rows[:15]
    plt.bar([r["edge"] for r in most_common], [r["state_fraction"] for r in most_common])
    plt.ylabel("Fraction of successful circuits using pair")
    plt.xlabel("Qubit pair")
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.savefig(args.output_dir / "common_qubit_pairs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    subset = [r for r in by_lag if r["trading_day_gap"] <= 60]
    plt.plot([r["trading_day_gap"] for r in subset],
             [r["mean_ordered_similarity"] for r in subset], marker=".")
    plt.axhline(global_mean, ls="--", color="gray", label="All-pairs mean")
    plt.xlabel("Separation (trading days)")
    plt.ylabel("Mean ordered topology similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "topology_similarity_vs_time.png", dpi=180)
    plt.close()

    print("Results written to", args.output_dir.resolve())
    print("Adjacent similarity:", summary["mean_adjacent_ordered_similarity"])
    print("All-pairs similarity:", summary["mean_global_ordered_similarity"])
    print("Structural reference topologies (not transfer-tested):", len(structural))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "analyse"))
    p.add_argument("--year", type=int, default=2013, choices=(2013, 2020))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--blocks", type=int, default=15)
    p.add_argument("--sweeps", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--min-fidelity", type=float, default=0.995)
    p.add_argument("--structural-threshold", type=float, default=0.8)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--exclude-degenerate", action="store_true")
    p.add_argument("--no-verify-fft", action="store_true")
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--halfwidth", type=float, default=None)
    args = p.parse_args()
    args.output_dir = args.output_dir or Path(f"aqce_results_{args.year}")
    args.n_fft = args.n_fft or (131072 if args.year == 2020 else 16384)
    args.halfwidth = args.halfwidth or (25.0 if args.year == 2020 else 3.5)
    if args.blocks < 1 or args.sweeps < 0 or args.window < 2 or args.stride < 1:
        p.error("Require --blocks>=1, --sweeps>=0, --window>=2, --stride>=1")
    if not 0 <= args.min_fidelity <= 1 or not 0 < args.structural_threshold <= 1:
        p.error("Fidelity and similarity thresholds must lie in [0,1]")
    if args.command == "run":
        run(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()
