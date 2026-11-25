"""Run reproducible synthetic Heston sweeps from the repository root."""
import argparse
import csv
from datetime import datetime, timezone
from itertools import combinations, product
import json
from pathlib import Path
import sys
import numpy as np

from .states import Coordinates, make_state
from .analysis import analyse, entropy

AXES = ("a", "b", "c", "d", "rho")
RANGES = {"a": (.01, 100.), "b": (.001, 10.), "c": (.1, 10.),
          "d": (.0025, .25), "rho": (-.99, .99)}
REFERENCE = (1., .1, 1., .04, -.7)


def grids(points, reference):
    out = {}
    for i, key in enumerate(AXES):
        lo, hi = RANGES[key]
        vals = np.linspace(lo, hi, points) if key == "rho" else np.geomspace(lo, hi, points)
        anchors = [reference[i]] + ({"a": [1.], "b": [1.], "rho": [0.]}.get(key, []))
        # Avoid near-duplicate floating point values, retaining exact anchors.
        for v in anchors:
            vals = vals[~np.isclose(vals, v, rtol=1e-12, atol=1e-15)]
            vals = np.r_[vals, v]
        out[key] = sorted(float(v) for v in vals)
    return out


def sample_models(grid, reference, vary, mode):
    groups = ([tuple(vary)] if mode == "full" or len(vary) == 1
              else list(combinations(vary, 2)))
    seen = set()
    for group in groups:
        for values in product(*(grid[key] for key in group)):
            case = list(reference)
            for key, value in zip(group, values):
                case[AXES.index(key)] = value
            case = tuple(case)
            if case not in seen:
                seen.add(case)
                yield case


def json_cell(value):
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (tuple, list, dict)) else value


def csv_writer(path, columns):
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    return f, writer


def plot_maps(out, grid, reference, vary, mode, records, epsilons, all_cuts):
    """Pairwise slices: other three coordinates exactly at reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    if len(vary) < 2:
        return
    for x, y in combinations(vary, 2):
        other = [k for k in AXES if k not in (x, y)]
        lookup = {(r[x], r[y], r["epsilon"]): r for r in records
                  if all(r[k] == reference[AXES.index(k)] for k in other)}
        xx, yy = np.array(grid[x]), np.array(grid[y])
        def edges(z, logarithmic):
            t = np.log(z) if logarithmic else z
            e = np.r_[t[0]-(t[1]-t[0])/2, (t[1:]+t[:-1])/2,
                      t[-1]+(t[-1]-t[-2])/2]
            return np.exp(e) if logarithmic else e
        for eps in epsilons:
            fields = [("natural_cut_bound", "Natural cut lower bound"),
                      ("natural_tt_cap", "Natural TT-SVD cap")]
            if all_cuts:
                fields += [("best_cut_bound", "Best-order cut lower bound"),
                           ("best_tested_tt_cap", "Best tested TT-SVD cap")]
            fig, axes = plt.subplots(1, len(fields), figsize=(4.1*len(fields), 3.8),
                                     constrained_layout=True, squeeze=False)
            matrices = [np.array([[lookup.get((u,v,eps), {}).get(field, np.nan)
                                    for u in xx] for v in yy], dtype=float)
                        for field, _ in fields]
            valid = np.concatenate([m[np.isfinite(m)] for m in matrices])
            top = int(valid.max()) if valid.size else 1
            cmap = plt.get_cmap("viridis", top).copy()
            cmap.set_bad("#dddddd")
            norm = BoundaryNorm(np.arange(.5, top+1.5), top)
            for ax, (field, label), matrix in zip(axes[0], fields, matrices):
                ax.set_facecolor("#dddddd")
                image = ax.pcolormesh(edges(xx, x!="rho"), edges(yy, y!="rho"),
                                      np.ma.masked_invalid(matrix), cmap=cmap, norm=norm)
                ax.set_xscale("linear" if x=="rho" else "log")
                ax.set_yscale("linear" if y=="rho" else "log")
                ax.set(xlabel=x, ylabel=y, title=label)
                ax.plot(reference[AXES.index(x)], reference[AXES.index(y)], "rx", ms=6)
            fig.colorbar(image, ax=axes[0].tolist(), ticks=np.arange(1, top+1))
            fig.suptitle(f"{x} vs {y} | infidelity <= {eps:g} | other coordinates fixed")
            fig.savefig(figures / f"{x}_{y}_eps_{eps:.0e}.png", dpi=160)
            plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=int, default=13, help="Grid points per axis, plus anchors")
    p.add_argument("--vary", nargs="+", choices=AXES, default=list(AXES))
    p.add_argument("--mode", choices=("pairwise","full"), default="pairwise")
    p.add_argument("--reference", type=float, nargs=5, default=REFERENCE, metavar=("A","B","C","D","RHO"))
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--maturity", type=float, default=1.)
    p.add_argument("--cutoff", type=float, default=6.)
    p.add_argument("--eps", nargs="+", type=float, default=[1e-2,1e-3,1e-4,1e-5,1e-6])
    p.add_argument("--n-fft", type=int, default=16384)
    p.add_argument("--max-fft", type=int, default=1048576)
    p.add_argument("--numerical-tol", type=float, default=1e-8)
    p.add_argument("--moment-tol", type=float, default=2e-4)
    p.add_argument("--natural-only", action="store_true")
    p.add_argument("--max-models", type=int, default=20000)
    p.add_argument("--name", help="Unique run subdirectory under regimes/results")
    p.add_argument("--no-states", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    ref = tuple(args.reference)
    Coordinates(*ref).parameters(args.maturity)
    if len(set(args.vary)) != len(args.vary):
        raise ValueError("--vary must not repeat axes")
    if args.points < 2 or not 2 <= args.qubits <= 12 or args.cutoff <= 0:
        raise ValueError("Require points >= 2, 2 <= qubits <= 12, cutoff > 0")
    if args.qubits > 10 and not args.natural_only:
        raise ValueError("For >10 qubits use --natural-only (all-cut SVDs grow exponentially)")
    if any(not 0 < e < 1 for e in args.eps) or not 0 < args.numerical_tol < 1 or not 0 < args.moment_tol < 1:
        raise ValueError("Tolerances must be strictly between zero and one")
    for n in (args.n_fft, args.max_fft):
        if n < 256 or n & (n-1):
            raise ValueError("FFT sizes must be powers of two >= 256")
    if 4*args.n_fft > args.max_fft:
        raise ValueError("max-fft must be at least 4*n-fft for verification")
    if args.numerical_tol > min(args.eps)/10:
        raise ValueError("Use numerical-tol <= smallest eps / 10")
    args.eps = sorted(set(args.eps), reverse=True)
    grid = grids(args.points, ref)
    # Bound the Cartesian product before constructing it in memory.
    if args.mode == "full" and np.prod([len(grid[k]) for k in args.vary], dtype=np.int64) > args.max_models:
        raise ValueError("Full grid exceeds --max-models; reduce points or vary fewer axes")
    cases = list(sample_models(grid, ref, args.vary, args.mode))
    if len(cases) > args.max_models:
        raise ValueError(f"{len(cases)} models exceeds --max-models")
    name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(name).name != name or name in (".","..",""):
        raise ValueError("--name must be a single new directory name")
    out = Path(__file__).resolve().parent / "results" / name
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), axes=AXES, grids=grid, model_count=len(cases),
                  all_cuts=not args.natural_only, numpy_version=np.__version__,
                  python_version=sys.version)
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model_cols = ["case_id", *AXES, "v0","kappa","theta","xi","status","error",
                  "mean","std","integrated_variance_mean","integrated_variance_cv2",
                  "retained_mass","fft_points","fft_halfwidth","resolution_infidelity",
                  "domain_infidelity","retained_mass_difference","moment_error",
                  "fft_mean_z","fft_second_moment_z","negative_mass","edge_mass"]
    rank_cols = ["case_id", *AXES,"epsilon","natural_cut_bound","natural_profile",
                 "natural_tt_cap","natural_tt_fidelity","best_cut_bound","best_cut_order",
                 "reordered_tt_cap","reordered_tt_fidelity","best_tested_tt_cap","cut_order_gap"]
    mf,mw = csv_writer(out/"models.csv",model_cols)
    rf,rw = csv_writer(out/"ranks.csv",rank_cols)
    sf,sw = csv_writer(out/"schmidt_spectra.csv",
                       ["case_id","mask","qubits","natural_cut","weights","entropy_bits"])
    ff,fw = csv_writer(out/"mps_fidelities.csv",["case_id","order","bond_cap","fidelity"])
    successes, records = 0, []
    ids, states = [], []
    try:
        for i, case in enumerate(cases):
            coords = Coordinates(*case)
            p = coords.parameters(args.maturity)
            base = dict(case_id=i, **dict(zip(AXES, case)))
            model = dict(base, v0=p.v0, kappa=p.kappa, theta=p.theta, xi=p.xi)
            try:
                prob, diag = make_state(coords, n_qubits=args.qubits, maturity=args.maturity,
                    cutoff=args.cutoff, n_fft=args.n_fft, max_fft=args.max_fft,
                    numerical_tol=args.numerical_tol, moment_tol=args.moment_tol)
                weights, curves, rows = analyse(np.sqrt(prob), args.eps, not args.natural_only)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                mw.writerow(dict(model, status="failed", error=str(exc)))
                mf.flush()
                print(f"[{i+1}/{len(cases)}] failed: {exc}", flush=True)
                continue
            mw.writerow(dict(model, status="ok", **diag))
            for mask, w in weights.items():
                sw.writerow(dict(case_id=i, mask=mask,
                    qubits=json_cell([q for q in range(args.qubits) if mask & (1<<q)]),
                    natural_cut=mask in {(1<<k)-1 for k in range(1,args.qubits)},
                    weights=json_cell(w.tolist()), entropy_bits=entropy(w)))
            for order, curve in curves.items():
                for cap, fidelity in enumerate(curve, 1):
                    fw.writerow(dict(case_id=i, order=json_cell(order),
                                     bond_cap=cap, fidelity=float(fidelity)))
            for row in rows:
                r = dict(base, **row)
                rw.writerow({k:json_cell(v) for k,v in r.items()})
                records.append(r)
            if not args.no_states:
                ids.append(i)
                states.append(prob)
            successes += 1
            for handle in (mf,rf,sf,ff):
                handle.flush()
            if (i+1)%10 == 0 or i+1 == len(cases):
                print(f"[{i+1}/{len(cases)}] {successes} successful", flush=True)
    finally:
        for handle in (mf,rf,sf,ff):
            handle.close()
        if not args.no_states:
            probabilities = np.asarray(states).reshape(-1, 1<<args.qubits)
            np.savez_compressed(out/"states.npz", case_ids=np.asarray(ids,dtype=int),
                probabilities=probabilities, amplitudes=np.sqrt(probabilities),
                edges=np.linspace(-args.cutoff,args.cutoff,(1<<args.qubits)+1))
    (out/"summary.json").write_text(json.dumps(dict(total=len(cases), successful=successes,
        failed=len(cases)-successes),indent=2),encoding="utf-8")
    if not args.no_plots:
        plot_maps(out, grid, ref, args.vary, args.mode, records, args.eps, not args.natural_only)
    print(f"Results: {out} ({successes}/{len(cases)} succeeded)", flush=True)
    if not successes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
