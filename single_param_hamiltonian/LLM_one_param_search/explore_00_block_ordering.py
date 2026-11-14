"""Exploratory check: how expressive is a product of 6 odd-Y rotations on a
pair compared with a full SO(4) gate, and which chain direction/ordering is
best for the all-variable (MPS-like) limit?  Benchmarks: MPS D1 fidelity for
2013-01-02 is 0.996914; mean over 2013 is 0.997781 (14 CNOT)."""
import json, itertools, time
import numpy as np
from ansatz import fit
from paulis import odd_y_strings
from states import load_states

dates, S, _ = load_states(2013)
targets = S[[0, 125, 249]]           # three dates incl. a mid-year one
mps_d1 = {"2013-01-02": 0.9969138303504886}

def block(i, j, order):
    singles = {"a": odd_y_strings((i,)), "b": odd_y_strings((j,))}
    two = {l[i] + l[j]: l for l in odd_y_strings((i, j))}   # 'YX','YZ','XY','ZY'
    out = []
    for tok in order:
        if tok in singles: out += singles[tok]
        else: out.append(two[tok])
    return out

orders = {
    "a,b,YX,YZ,XY,ZY": ["a", "b", "YX", "YZ", "XY", "ZY"],
    "a,b,YZ,ZY,YX,XY": ["a", "b", "YZ", "ZY", "YX", "XY"],
    "a,b,YZ,XY,YX,ZY": ["a", "b", "YZ", "XY", "YX", "ZY"],
    "YZ,ZY,YX,XY,a,b": ["YZ", "ZY", "YX", "XY", "a", "b"],
    "a,YZ,b,ZY,a,YX,b": ["a", "YZ", "b", "ZY", "a", "YX", "b"],      # 8 rotations (repeats)
    "a,b,YZ,YX,a,b,ZY,XY,a,b": ["a", "b", "YZ", "YX", "a", "b", "ZY", "XY", "a", "b"],  # 12 rotations
}
results = {}
for direction in ("msb_first", "lsb_first"):
    pairs = [(i, i + 1) for i in range(7)]
    if direction == "lsb_first": pairs = pairs[::-1]
    for name, order in orders.items():
        labels = sum([block(i, j, order) for i, j in pairs], [])
        t = time.time()
        r = fit(labels, targets, variable=labels, restarts=6, maxiter=1500, seed=1, init_scale=0.4)
        results[f"{direction}|{name}"] = {"K": len(labels), "F": r.fidelities.tolist(), "s": time.time() - t}
        print(f"{direction:9s} {name:28s} K={len(labels):3d}  F={np.round(r.fidelities,5)}  {time.time()-t:.1f}s")
json.dump(results, open("results/explore_00_block_ordering.json", "w"), indent=1)
