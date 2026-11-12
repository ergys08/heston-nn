"""Rankings, frontier, method comparison and plots from the evaluation DB.

    python -m topology_search_2013.report
"""
import json
from pathlib import Path

import numpy as np

from .evaluator import Store, rank_key, stage_signature
from .search import CONFIG, RESULTS, full_stage

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def family(source):
    s = source or ""
    if s.startswith(("llm", "prior:llm")):
        return "llm"
    if s.startswith(("evo", "prior:evo")):
        return "evolutionary"
    if s.startswith("local"):
        return "local"
    if s.startswith("random"):
        return "random"
    if "aqce" in s:
        return "aqce_seed"
    return "other"


def write_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(json.dumps(r.get(c)) if isinstance(r.get(c), (list, dict)) or
                             (isinstance(r.get(c), str) and "," in r.get(c)) else str(r.get(c))
                             for c in cols) + "\n")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    store = Store(RESULTS / "evaluations.sqlite")
    fs = full_stage()
    sig = stage_signature(fs)
    rows = store.query("SELECT * FROM evals WHERE stage_sig=?", (sig,))
    rows.sort(key=rank_key, reverse=True)
    cols = ["M", "depth", "coverage", "min", "p5", "median", "mean", "block_updates",
            "seconds", "source", "topology"]
    write_csv(RESULTS / "ranking_full_stage.csv", rows, cols)

    # frontier: best per M at full stage, overridden by the final stage if present
    finals = store.query("SELECT * FROM evals WHERE stage='final'")
    frontier = []
    for M in sorted({r["M"] for r in rows}):
        best = max((r for r in rows if r["M"] == M), key=rank_key)
        fin = [r for r in finals if r["M"] == M]
        fbest = max(fin, key=rank_key) if fin else None
        frontier.append(dict(M=M, full_coverage=best["coverage"], full_min=best["min"],
                             full_mean=best["mean"], full_topology=best["topology"],
                             full_source=best["source"],
                             n_evaluated=sum(r["M"] == M for r in rows),
                             final_coverage=fbest and fbest["coverage"],
                             final_min=fbest and fbest["min"], final_mean=fbest and fbest["mean"],
                             final_topology=fbest and fbest["topology"]))
    write_csv(RESULTS / "frontier.csv", frontier, list(frontier[0]))

    # method comparison (all stages count toward budget; quality at full stage)
    allrows = store.query("SELECT M, source, block_updates, stage_sig, coverage, min, mean, n_states, key, created FROM evals")
    comp = []
    for M in sorted({r["M"] for r in rows}):
        for fam in ("llm", "evolutionary", "local", "random", "aqce_seed"):
            sub = [r for r in allrows if r["M"] == M and family(r["source"]) == fam]
            full = [r for r in sub if r["stage_sig"] == sig]
            if not sub:
                continue
            b = max(full, key=lambda r: rank_key(dict(r, M=M))) if full else None
            comp.append(dict(M=M, method=fam, candidates=len({r["key"] for r in sub}),
                             full_evals=len(full),
                             block_updates=int(sum(r["block_updates"] for r in sub)),
                             best_coverage=b and b["coverage"], best_min=b and b["min"],
                             full_coverage_fraction=(np.mean([r["coverage"] == 250 for r in full])
                                                     if full else None)))
    write_csv(RESULTS / "method_comparison.csv", comp, list(comp[0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. frontier: 1 - min fidelity vs blocks (log), coverage annotated
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax)
    Ms = [f["M"] for f in frontier]
    mins = [f["final_min"] or f["full_min"] for f in frontier]
    means = [f["final_mean"] or f["full_mean"] for f in frontier]
    ax.plot(Ms, 1 - np.array(mins), "-o", color=BLUE, lw=2, ms=6, label="worst state (1 − min F)")
    ax.plot(Ms, 1 - np.array(means), "-o", color=ORANGE, lw=2, ms=6, label="average (1 − mean F)")
    ax.axhline(1 - CONFIG["threshold"], color=INK2, ls="--", lw=1)
    ax.text(Ms[-1], 1 - CONFIG["threshold"], " F = 0.995", color=INK2, va="bottom", ha="right", fontsize=9)
    for f, m in zip(frontier, mins):
        cov = f["final_coverage"] or f["full_coverage"]
        ax.annotate(f"{cov}/250", (f["M"], 1 - m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Two-qubit blocks (M)", color=INK)
    ax.set_ylabel("Infidelity", color=INK)
    ax.set_title("Best shared topology per block budget (all 250 states of 2013)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "frontier.png", dpi=180)
    plt.close(fig)

    # 2. method comparison: best-so-far min fidelity vs cumulative budget at the headline M
    for M in sorted({c["M"] for c in comp if c["method"] in ("random", "evolutionary", "llm")}):
        fig, ax = plt.subplots(figsize=(7, 4))
        style(ax)
        drawn = 0
        for fam, color in (("evolutionary", BLUE), ("llm", ORANGE), ("random", AQUA), ("aqce_seed", YELLOW)):
            sub = sorted([r for r in allrows if r["M"] == M and family(r["source"]) == fam],
                         key=lambda r: r["created"])
            if not sub:
                continue
            budget, best, xs, ys = 0, None, [], []
            for r in sub:
                budget += r["block_updates"]
                if r["stage_sig"] == sig:
                    k = (r["coverage"], r["min"])
                    best = k if best is None or k > best else best
                if best:
                    xs.append(budget); ys.append(best[1] if best[0] == 250 else np.nan)
            if xs:
                ax.step(xs, ys, where="post", color=color, lw=2, label=fam)
                drawn += 1
        if drawn < 2:
            plt.close(fig)
            continue
        ax.axhline(CONFIG["threshold"], color=INK2, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Cumulative evaluation budget (single-block SVD updates)", color=INK)
        ax.set_ylabel("Best min fidelity with 250/250 coverage", color=INK)
        ax.set_title(f"Search methods at M = {M} (gaps: no full-coverage topology yet)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS / f"methods_M{M:02d}.png", dpi=180)
        plt.close(fig)
    print("frontier:")
    for f in frontier:
        print(f"  M={f['M']:2d} full: cov={f['full_coverage']} min={f['full_min']:.6f} "
              f"| final: cov={f['final_coverage']} min={f['final_min']} | {f['full_topology']}")
    for c in comp:
        print(" ", c)


if __name__ == "__main__":
    main()
