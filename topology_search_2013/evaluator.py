"""Staged, cached, parallel evaluation of candidate topologies.

Every completed evaluation is written immediately to an SQLite database
(``results/evaluations.sqlite``), keyed by the topology's commutation
normal form and the stage signature, so an interrupted search resumes
without repeating work and the evaluation budget of each candidate is
recorded (block_updates = number of single-block SVD updates, seconds =
worker wall time).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import topology as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGETS_FILE = ROOT / "aqce_results_2013" / "target_states.npz"
THRESHOLD = 0.995

_TARGETS = None


def load_targets():
    global _TARGETS
    if _TARGETS is None:
        with np.load(TARGETS_FILE, allow_pickle=False) as d:
            _TARGETS = (d["dates"].tolist(), d["amplitudes"].copy())
    return _TARGETS


def stage_signature(stage):
    """Short hash of everything that affects an evaluation's outcome."""
    s = json.dumps({k: stage[k] for k in sorted(stage) if k != "name"}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def summarise(f, threshold=THRESHOLD):
    f = np.asarray(f)
    return dict(coverage=int(np.sum(f >= threshold)), n_states=len(f),
                min=float(f.min()), p5=float(np.percentile(f, 5)),
                median=float(np.median(f)), mean=float(f.mean()))


def rank_key(row):
    """Strict priority: coverage, then minimum, then mean, then fewer blocks."""
    return (row["coverage"] / row["n_states"], row["min"], row["mean"], -row["M"])


# --------------------------------------------------------------------- worker

def _worker_init():
    import torch
    torch.set_num_threads(1)
    load_targets()


def _evaluate(job):
    from circuits.fixed_topology_torch import optimise
    import torch
    top, stage = job
    _, amps = load_targets()
    idx = stage["states"]
    t0 = time.perf_counter()
    dtype = torch.complex128 if stage.get("complex") else torch.float64
    res = optimise(amps[idx], list(top), restarts=stage["restarts"],
                   sweeps=stage["sweeps"], dtype=dtype, seed=stage.get("seed", 0),
                   stop_at=stage.get("stop_at"), tol=stage.get("tol", 1e-9),
                   window=stage.get("window", 10),
                   prescreen_sweeps=stage.get("prescreen_sweeps", 0), keep=stage.get("keep"))
    out = dict(fidelities=res.fidelity.tolist(), block_updates=int(res.block_updates),
               sweeps_run=int(res.sweeps_run), seconds=time.perf_counter() - t0)
    if stage.get("keep_unitaries"):
        out["unitaries"] = res.unitaries
        out["restart_fidelity"] = res.restart_fidelity
    return out


# ---------------------------------------------------------------------- cache

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS evals (
            key TEXT, stage TEXT, stage_sig TEXT, topology TEXT, M INTEGER,
            depth INTEGER, n_states INTEGER, coverage INTEGER, min REAL,
            p5 REAL, median REAL, mean REAL, fidelities TEXT,
            block_updates INTEGER, seconds REAL, sweeps_run INTEGER,
            source TEXT, generation INTEGER, created REAL,
            PRIMARY KEY (key, stage_sig))""")
        self.db.commit()

    def get(self, k, sig):
        cur = self.db.execute("SELECT * FROM evals WHERE key=? AND stage_sig=?", (k, sig))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([c[0] for c in cur.description], row))

    def put(self, row):
        cols = list(row)
        self.db.execute(
            f"INSERT OR REPLACE INTO evals ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols])
        self.db.commit()

    def query(self, sql, args=()):
        cur = self.db.execute(sql, args)
        names = [c[0] for c in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


class Evaluator:
    def __init__(self, store, workers=None, log=None):
        self.store = store
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.pool = ProcessPoolExecutor(self.workers, initializer=_worker_init)
        self.log = log or print

    def close(self):
        self.pool.shutdown()

    def evaluate(self, tops, stage, sources=None, generation=None):
        """Evaluate topologies at one stage (cached). Returns rows in order."""
        sig = stage_signature(stage)
        tops = [T.normal_form(t) for t in tops]
        sources = sources or ["?"] * len(tops)
        rows, todo, seen = [None] * len(tops), [], {}
        for i, t in enumerate(tops):
            k = T.key(t)
            hit = self.store.get(k, sig)
            if hit is not None:
                rows[i] = hit
            elif k in seen:
                seen[k].append(i)
            else:
                seen[k] = [i]
                todo.append(i)
        futures = {i: self.pool.submit(_evaluate, (tops[i], stage)) for i in todo}
        for i, fut in futures.items():
            out = fut.result()
            f = out["fidelities"]
            row = dict(key=T.key(tops[i]), stage=stage["name"], stage_sig=sig,
                       topology=json.dumps([list(p) for p in tops[i]]),
                       M=len(tops[i]), depth=T.depth(tops[i]), **summarise(f),
                       fidelities=json.dumps([round(x, 10) for x in f]),
                       block_updates=out["block_updates"], seconds=out["seconds"],
                       sweeps_run=out["sweeps_run"], source=sources[i],
                       generation=generation, created=time.time())
            self.store.put(row)
            for j in seen[T.key(tops[i])]:
                rows[j] = row
        return rows
