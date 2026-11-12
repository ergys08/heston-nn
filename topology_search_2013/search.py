"""Persistent, resumable topology search for the 250 historical 2013 states.

Run from the repository root (all commands resume from the SQLite cache and
results/state.json; interrupting at any time loses at most the evaluations
in flight):

    python -m topology_search_2013.search seed
    python -m topology_search_2013.search evolve --M 8 --generations 20
    python -m topology_search_2013.search random --M 8 --budget 400
    python -m topology_search_2013.search final --M 8 9 10
    python -m topology_search_2013.search report

Ranking is strictly lexicographic: coverage (F >= 0.995), then minimum
fidelity, then mean fidelity; block count is handled by searching each M
separately and reporting the frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from . import topology as T
from .evaluator import Evaluator, Store, load_targets, rank_key, stage_signature

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
T_ROOT = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text())
LOG = RESULTS / "log.jsonl"
STATE = RESULTS / "state.json"


def log(event, **kw):
    RESULTS.mkdir(exist_ok=True)
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), event=event, **kw)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:400], flush=True)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def full_stage():
    return dict(CONFIG["stages"]["full"], name="full", states=list(range(250)))


def screen_stage(subset):
    return dict(CONFIG["stages"]["screen"], name="screen", states=sorted(subset))


def ranked(store, stage, M=None, limit=None):
    sig = stage_signature(stage)
    rows = store.query("SELECT * FROM evals WHERE stage_sig=?" + (" AND M=?" if M else ""),
                       (sig, M) if M else (sig,))
    rows.sort(key=rank_key, reverse=True)
    return rows[:limit] if limit else rows


def top_of(row):
    return tuple(tuple(p) for p in json.loads(row["topology"]))


# ----------------------------------------------------------- hard subsets

def hard_subset(elite_rows, size_hard, size_diverse, seed=0):
    """Hardest states for the current elites + farthest-point diverse states."""
    f = np.array([json.loads(r["fidelities"]) for r in elite_rows])
    score = np.mean(np.argsort(np.argsort(f, axis=1), axis=1), axis=0)  # mean rank
    score += 1000 * np.mean(f < 0.995, axis=0) * -1                      # failing first
    chosen = list(np.argsort(score)[:size_hard])
    _, amps = load_targets()
    overlap = np.abs(amps @ amps.T) ** 2
    while len(chosen) < size_hard + size_diverse:
        d = 1 - overlap[:, chosen].max(axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return sorted(int(i) for i in chosen)


# --------------------------------------------------------------- children

def edge_weights(rows):
    w = {p: 1.0 for p in T.ALL_PAIRS}
    for r in rows:
        for p in top_of(r):
            w[p] += 1.0
    return w


def make_children(elites, neighbours, M, n_children, rng, weights):
    """Mutations / recombinations of same-M elites, plus pruning of M+1
    and growth of M-1 elites (from ``neighbours``)."""
    ops = CONFIG["operators"]
    names, probs = zip(*ops.items())
    children, sources, seen = [], [], {T.key(top_of(e)) for e in elites}
    tries = 0
    while len(children) < n_children and tries < 50 * n_children:
        tries += 1
        op = rng.choices(names, weights=probs)[0]
        a = top_of(rng.choice(elites[: max(2, len(elites))]))
        if op in T.MUTATIONS:
            child = a
            for _ in range(1 + (rng.random() < 0.3) + (rng.random() < 0.1)):
                child = T.MUTATIONS[op](child, rng, weights)
                op2 = rng.choice(list(T.MUTATIONS))
                if rng.random() < 0.2:
                    child = T.MUTATIONS[op2](child, rng, weights)
        elif op in T.CROSSOVERS:
            b = top_of(rng.choice(elites))
            child = T.CROSSOVERS[op](a, b, rng)
        elif op == "prune" and neighbours.get(M + 1):
            child = list(top_of(rng.choice(neighbours[M + 1])))
            del child[rng.randrange(len(child))]
        elif op == "grow" and neighbours.get(M - 1):
            child = T.mut_insert(top_of(rng.choice(neighbours[M - 1])), rng, weights)
        elif op == "random":
            child = T.random_topology(M, rng, weights)
        else:
            continue
        child = T.repair(child, M, rng, weights)
        if child is None:
            continue
        k = T.key(child)
        if k in seen:
            continue
        seen.add(k)
        children.append(child)
        sources.append(f"evo:{op}")
    return children, sources


# ---------------------------------------------------------------- commands

def cmd_evolve(args, ev, store):
    state = load_state()
    rng = random.Random()
    for M in args.M:
        mstate = state.setdefault(f"M{M}", {"generation": 0})
        if "rng" in mstate:
            rng.setstate(tuple(_tuplify(mstate["rng"])))
        else:
            rng.seed(CONFIG["seed"] * 1000 + M)
        fs = full_stage()
        for g in range(args.generations):
            elites = ranked(store, fs, M, CONFIG["elite_size"])
            if len(elites) < 2:
                log("evolve_skip", M=M, reason="fewer than 2 full-stage evaluations at this M; run seed")
                break
            gen = mstate["generation"]
            if "subset" not in mstate or gen % CONFIG["resubset_every"] == 0:
                mstate["subset"] = hard_subset(elites, CONFIG["screen_hard"], CONFIG["screen_diverse"])
            ss = screen_stage(mstate["subset"])
            neighbours = {m: ranked(store, fs, m, CONFIG["elite_size"]) for m in (M - 1, M + 1)}
            weights = edge_weights(elites)
            children, sources = make_children(elites, neighbours, M, args.children, rng, weights)
            # screen elites too so that screen scores are comparable
            elite_screen = ev.evaluate([top_of(e) for e in elites], ss, ["elite"] * len(elites), gen)
            scr = ev.evaluate(children, ss, sources, gen)
            bar = sorted(elite_screen, key=rank_key, reverse=True)[min(len(elite_screen), CONFIG["promote_bar_rank"]) - 1]
            order = sorted(range(len(children)), key=lambda i: rank_key(scr[i]), reverse=True)
            promote = [i for i in order if rank_key(scr[i])[:2] >= rank_key(bar)[:2]][: args.promote]
            if len(promote) < CONFIG["min_promote"]:
                promote = order[: CONFIG["min_promote"]]
            full = ev.evaluate([children[i] for i in promote], fs, [sources[i] for i in promote], gen)
            new_elites = ranked(store, fs, M, CONFIG["elite_size"])
            best = new_elites[0]
            mstate["generation"] = gen + 1
            mstate["rng"] = rng.getstate()
            mstate["best"] = dict(topology=best["topology"], coverage=best["coverage"],
                                  min=best["min"], mean=best["mean"])
            save_state(state)
            log("generation", M=M, generation=gen, children=len(children), promoted=len(promote),
                best_coverage=best["coverage"], best_min=round(best["min"], 6),
                best_mean=round(best["mean"], 6), best_topology=best["topology"],
                promoted_best_cov=max([r["coverage"] for r in full], default=None),
                screen_subset_size=len(mstate["subset"]))


def _tuplify(x):
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


def cmd_random(args, ev, store):
    """Random-search baseline: uniform random valid topologies, full stage."""
    fs = full_stage()
    for M in args.M:
        rng = random.Random(CONFIG["seed"] * 7 + M)
        tops, seen = [], set()
        while len(tops) < args.budget:
            t = T.random_topology(M, rng)
            if T.key(t) not in seen:
                seen.add(T.key(t)); tops.append(t)
        rows = ev.evaluate(tops, fs, ["random"] * len(tops), generation=-2)
        best = max(rows, key=rank_key)
        log("random_baseline", M=M, n=len(rows), best_coverage=best["coverage"],
            best_min=best["min"], best_topology=best["topology"],
            coverage_hist=np.bincount([r["coverage"] for r in rows], minlength=251)[-10:].tolist())


def aqce_topologies():
    from circuits.fixed_topology import pairs_from_qulacs
    out = {}
    for folder in ("aqce_results_2013", "aqce_results_2013_M20"):
        for f in sorted((T_ROOT / folder / "unitaries").glob("*.npz")):
            with np.load(f) as d:
                out[f"{folder}:{f.stem}"] = T.canon_pairs(pairs_from_qulacs(d["pairs"]))
    return out


def structured_families(M):
    """Simple hand-written families (nearest-neighbour staircases/bricks)."""
    desc = [(q, q + 1) for q in reversed(range(7))]
    asc = [(q, q + 1) for q in range(7)]
    brick = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)]
    fams = {"stair_desc": desc, "stair_asc": asc, "stair_zigzag": desc + asc,
            "brick": brick}
    out = {}
    for name, unit in fams.items():
        t = (unit * (M // len(unit) + 2))[:M]
        if T.valid(t):
            out[f"family:{name}"] = tuple(t)
    return out


def cmd_seed(args, ev, store):
    """Seed every M with: suffixes of all 500 saved AQCE circuits (the
    blocks AQCE inserted first), the per-state AQCE topologies of matching
    length, structured families, and the best earlier evaluations."""
    fs = full_stage()
    aq = aqce_topologies()
    for M in args.M:
        cands = {}
        for name, t in aq.items():
            s = T.simplify(t[-M:]) if len(t) >= M else None
            if s and len(s) == M and T.valid(s):
                cands.setdefault(T.key(s), (s, f"aqce_suffix:{name}"))
        for name, t in structured_families(M).items():
            cands.setdefault(T.key(t), (t, name))
        prior = store.query("SELECT topology, source FROM evals WHERE M=? AND stage_sig!=?",
                            (M, stage_signature(fs)))
        prior.sort(key=lambda r: 0)
        for r in prior:
            t = top_of(r)
            cands.setdefault(T.key(t), (t, f"prior:{r['source']}"))
        # bound the seeding cost: screen everything, promote the best
        items = list(cands.values())
        rows = store.query("SELECT key FROM evals WHERE stage_sig=? AND M=?", (stage_signature(fs), M))
        if len(rows) >= args.seed_keep:
            log("seed_skip", M=M, reason="already seeded")
            continue
        sub = hard_subset_from_reference(M)
        ss = screen_stage(sub)
        scr = ev.evaluate([t for t, _ in items], ss, [s for _, s in items], generation=0)
        order = sorted(range(len(items)), key=lambda i: rank_key(scr[i]), reverse=True)[: args.seed_keep]
        full = ev.evaluate([items[i][0] for i in order], fs, [items[i][1] for i in order], generation=0)
        best = max(full, key=rank_key)
        log("seed", M=M, candidates=len(items), promoted=len(order), best_coverage=best["coverage"],
            best_min=best["min"], best_source=best["source"], best_topology=best["topology"])


def hard_subset_from_reference(M):
    """Initial screening subset: hardest states for the 7-block staircase
    (M <= 10) or for the 15-block AQCE suffixes (M > 10), plus diverse states."""
    from circuits.fixed_topology_torch import optimise
    _, amps = load_targets()
    desc = [(q, q + 1) for q in reversed(range(7))]
    ref = desc if M <= 10 else desc * 2
    f = optimise(amps, ref, restarts=2, sweeps=200).fidelity
    fake = [dict(fidelities=json.dumps(f.tolist()))]
    return hard_subset(fake, CONFIG["screen_hard"], CONFIG["screen_diverse"])


def neighbourhood(top, M):
    """All single-edit neighbours of ``top`` that have exactly M blocks."""
    top = list(top)
    out = set()
    if len(top) == M:
        for i in range(M):
            for p in T.ALL_PAIRS:
                out.add(tuple(top[:i] + [p] + top[i + 1:]))
            g = top[i]
            rest = top[:i] + top[i + 1:]
            for j in range(M):
                out.add(tuple(rest[:j] + [g] + rest[j:]))
    elif len(top) == M + 1:
        for i in range(M + 1):
            out.add(tuple(top[:i] + top[i + 1:]))
    elif len(top) == M - 1:
        for i in range(M):
            for p in T.ALL_PAIRS:
                out.add(tuple(top[:i] + [p] + top[i:]))
    res = {}
    for t in out:
        if T.valid(t) and len(t) == M:
            res.setdefault(T.key(t), t)
    return list(res.values())


def cmd_local(args, ev, store):
    """Exhaustive 1-edit neighbourhood search (steepest ascent) around the
    top-k elites of M (replace/move) and of M+-1 (delete/insert)."""
    state = load_state()
    fs = full_stage()
    for M in args.M:
        mstate = state.setdefault(f"M{M}", {"generation": 0})
        for it in range(args.generations):
            elites = ranked(store, fs, M, CONFIG["elite_size"])
            if "subset" not in mstate:
                mstate["subset"] = hard_subset(elites, CONFIG["screen_hard"], CONFIG["screen_diverse"])
            ss = screen_stage(mstate["subset"])
            before = rank_key(elites[0])
            cands = {}
            for e in elites[: args.top]:
                for t in neighbourhood(top_of(e), M):
                    cands.setdefault(T.key(t), (t, "local:edit"))
            for m2 in (M - 1, M + 1):
                for e in ranked(store, fs, m2, 3):
                    for t in neighbourhood(top_of(e), M):
                        cands.setdefault(T.key(t), (t, "local:prune" if m2 > M else "local:grow"))
            items = list(cands.values())
            ev.evaluate([top_of(e) for e in elites], ss, ["elite"] * len(elites), -4)
            scr = ev.evaluate([t for t, _ in items], ss, [s for _, s in items], -4)
            order = sorted(range(len(items)), key=lambda i: rank_key(scr[i]), reverse=True)[: args.promote]
            ev.evaluate([items[i][0] for i in order], fs, [items[i][1] for i in order], -4)
            best = ranked(store, fs, M, 1)[0]
            log("local", M=M, iteration=it, neighbours=len(items), promoted=len(order),
                improved=rank_key(best) > before, best_coverage=best["coverage"],
                best_min=round(best["min"], 6), best_mean=round(best["mean"], 6),
                best_topology=best["topology"])
            save_state(state)
            if rank_key(best) <= before:
                break


def cmd_final(args, ev, store):
    """Heavy evaluation (complex unitaries, many restarts) of the top
    ``--top`` full-stage topologies per M; saves all optimised unitaries."""
    from .evaluator import _evaluate, summarise
    fs = full_stage()
    st = dict(CONFIG["stages"]["final"], name="final", states=list(range(250)),
              keep_unitaries=True)
    sig = stage_signature({k: v for k, v in st.items() if k != "keep_unitaries"})
    dates, amps = load_targets()
    jobs = []
    for M in args.M:
        for rank, row in enumerate(ranked(store, fs, M, args.top)):
            jobs.append((M, rank, row))
    if args.file:
        for name, t in json.loads(Path(args.file).read_text()).items():
            jobs.append((len(t), name, dict(topology=json.dumps(t), source=name)))
    futs = [(M, rank, row, ev.pool.submit(_evaluate, (top_of(row), st))) for M, rank, row in jobs]
    for M, rank, row, fut in futs:
        out = fut.result()
        top = T.normal_form(top_of(row))
        f = np.array(out["fidelities"])
        d = RESULTS / "final" / f"M{len(top):02d}_{hashlib.sha1(T.key(top).encode()).hexdigest()[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "unitaries.npz", unitaries=out["unitaries"],
                            pairs=np.array(top), fidelities=f,
                            restart_fidelity=out["restart_fidelity"])
        (d / "fidelities.csv").write_text("index,date,fidelity\n" + "".join(
            f"{i},{dates[i]},{f[i]:.12f}\n" for i in range(len(f))))
        summ = summarise(f)
        rec = dict(topology=[list(p) for p in top], M=len(top), depth=T.depth(top),
                   source=row.get("source"), full_stage_rank=rank, **summ,
                   block_updates=out["block_updates"], seconds=out["seconds"],
                   stage=st | {"states": "all 250"}, stage_sig=sig)
        rf = out["restart_fidelity"]
        rec["restart_sensitivity"] = dict(
            frac_restarts_within_1e5_of_best=float(np.mean(rf >= f[:, None] - 1e-5)),
            states_where_median_restart_fails=int(np.sum(np.median(rf, 1) < CONFIG["threshold"])),
            states_where_some_restart_fails=int(np.sum(rf.min(1) < CONFIG["threshold"])))
        (d / "summary.json").write_text(json.dumps(rec, indent=1))
        store.put(dict(key=T.key(top), stage="final", stage_sig=sig,
                       topology=json.dumps([list(p) for p in top]), M=len(top),
                       depth=T.depth(top), **summ,
                       fidelities=json.dumps([round(x, 12) for x in f.tolist()]),
                       block_updates=out["block_updates"], seconds=out["seconds"],
                       sweeps_run=out["sweeps_run"], source=str(row.get("source")),
                       generation=-5, created=time.time()))
        log("final", M=len(top), dir=d.name, coverage=summ["coverage"], min=summ["min"],
            mean=summ["mean"], topology=rec["topology"], seconds=round(out["seconds"]))


def cmd_eval(args, ev, store):
    """Evaluate topologies listed in a JSON file {name: [[a,b],...]}."""
    props = json.loads(Path(args.file).read_text())
    rows = ev.evaluate(list(props.values()), full_stage(), [f"{args.source}:{k}" for k in props], -3)
    for k, r in sorted(zip(props, rows), key=lambda x: rank_key(x[1]), reverse=True):
        print(f"{k:40s} M={r['M']:2d} cov={r['coverage']} min={r['min']:.6f} mean={r['mean']:.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("seed", "evolve", "random", "eval", "local", "final"))
    p.add_argument("--M", type=int, nargs="+", default=[8])
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--children", type=int, default=CONFIG["children"])
    p.add_argument("--promote", type=int, default=CONFIG["promote"])
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--file")
    p.add_argument("--seed-keep", type=int, default=40)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--source", default="llm")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    store = Store(RESULTS / "evaluations.sqlite")
    ev = Evaluator(store, workers=args.workers, log=print)
    try:
        {"seed": cmd_seed, "evolve": cmd_evolve, "local": cmd_local, "final": cmd_final, "random": cmd_random, "eval": cmd_eval}[args.command](args, ev, store)
    finally:
        ev.close()


if __name__ == "__main__":
    main()
