"""Reproduce 2013 all-to-all QMI and contiguous-cut measurements.

Run at repository root:
    python -m research.interaction_maps_2013.recompute
"""
from pathlib import Path
import csv
import json
import numpy as np

from neural_network.training_workflow import read_heston_states

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def entropy(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals, 0., None)
    vals /= vals.sum()
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log2(vals)))


def reduced_density(psi, qubits, n=8):
    # q0 = least-significant bit; NumPy tensor axis n-1-q.
    qubits = tuple(qubits)
    axes = [n-1-q for q in qubits]
    rest = [a for a in range(n) if a not in axes]
    m = np.transpose(psi.reshape((2,)*n), axes + rest).reshape(2**len(qubits), -1)
    return m @ m.conj().T


def main():
    dates, states = read_heston_states(REPO / 'heston/heston_params_2013.csv', num_qubits=8)
    pairs = [(i,j) for i in range(8) for j in range(i+1,8)]
    pair_rows, cut_rows = [], []
    for date, psi in zip(dates, states):
        psi = psi / np.linalg.norm(psi)
        singles = [entropy(reduced_density(psi, [i])) for i in range(8)]
        pair_values = [max(0., singles[i] + singles[j] - entropy(reduced_density(psi,[i,j])))
                       for i,j in pairs]
        cuts, fmax = [], []
        for k in range(1,8):
            qs = list(range(k)) if k <= 4 else list(range(k,8))
            vals = np.linalg.eigvalsh(reduced_density(psi, qs))
            vals = np.clip(vals, 0., None)
            vals /= vals.sum()
            good = vals[vals>1e-15]
            cuts.append(float(-2*np.sum(good*np.log2(good))))
            fmax.append(float(vals.max()))
        pair_rows.append([date,*pair_values])
        cut_rows.append([date,*cuts,*fmax])
    headers_pairs = ['date'] + [f'q{i+1}_q{j+1}' for i,j in pairs]
    headers_cut = ['date'] + [f'cut_after_q{k}' for k in range(1,8)] + [
        f'fmax_after_q{k}' for k in range(1,8)]
    for fn, header, data in [
        ('pairwise_qmi_daily_recomputed.csv',headers_pairs,pair_rows),
        ('cuts_daily_recomputed.csv',headers_cut,cut_rows),
    ]:
        with (HERE/fn).open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(header)
            w.writerows(data)
    print('Recomputed', len(dates), 'dates')
    print('Mean all-pairs QMI (bits):')
    means = np.asarray([r[1:] for r in pair_rows]).mean(axis=0)
    for (i,j), val in sorted(zip(pairs, means),key=lambda x:-x[1])[:10]:
        print(f'  q{i+1}-q{j+1}: {val:.6f}')


if __name__ == '__main__':
    main()
