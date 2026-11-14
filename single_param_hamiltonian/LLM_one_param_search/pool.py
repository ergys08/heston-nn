"""Candidate pool bookkeeping: every evaluated architecture is appended to
results/pool.jsonl; discarded ones are noted in results/discard_log.md."""
from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path
import numpy as np

from paulis import compact_label

HERE = Path(__file__).resolve().parent
POOL = HERE / "results" / "pool.jsonl"
DISCARD = HERE / "results" / "discard_log.md"


def arch_id(labels, variable):
    h = hashlib.sha1(("|".join(labels) + "#" + "|".join(sorted(variable))).encode()).hexdigest()[:10]
    return h


def record(stage, labels, variable, train_indices, fit_result=None, cx=None, notes="", extra=None, dof=None):
    labels = list(labels); variable = [int(v) for v in variable]   # positions
    entry = {
        "id": arch_id(labels, [str(v) for v in variable]), "stage": stage, "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels": labels, "labels_compact": [compact_label(l) for l in labels],
        "variable_positions": variable, "variable_labels": [compact_label(labels[v]) for v in variable],
        "n_terms": len(labels), "n_variable": len(variable), "n_dof": len(variable) if dof is None else dof,
        "train_indices": list(map(int, train_indices)), "n_train": len(train_indices), "notes": notes,
    }
    if cx: entry.update(cx)
    if fit_result is not None:
        entry.update({"mean_F": fit_result.mean_fidelity, "min_F": fit_result.min_fidelity,
                      "fidelities": [float(x) for x in fit_result.fidelities],
                      "coefficients": fit_result.coefficients_dict()})
    if extra: entry.update(extra)
    with POOL.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def discard(arch, reason):
    with DISCARD.open("a") as f:
        f.write(f"- `{arch}`: {reason}\n")


def load_pool():
    if not POOL.exists():
        return []
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]
