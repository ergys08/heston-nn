# Heston 2013: all-pairs QMI and MPS cuts

Analysed 250 dated, 8-qubit terminal log-return amplitude-encoded Heston targets from `heston/heston_params_2013.csv`, 2013-01-02 to 2013-12-31.

A numerical JavaScript implementation of `heston/distributions.py`'s Heston characteristic function, 16384-point Fourier FFT inversion and 256-bin CDF interpolation was used, with maturity=1, drift=.05, halfwidth=3.5 and cutoff=2.5. The first historical state was independently cross-checked against the NumPy original at selected amplitudes. A Python reproducibility script can recompute all values using the repository's native Python generator.

Qubits use little-endian indexing: q1 is the least significant bit, q8 the most significant. `pairwise_qmi_daily.csv` contains all 28 pairwise QMI values (bits) per date. `cuts_daily.csv` contains all seven contiguous bipartition QMI values (2 times von Neumann entropy, in bits) and the largest squared Schmidt coefficient: an upper bound on fidelity for a product state across that partition. `mean_pairwise_matrix.csv` is the averaged QMI matrix. `summary.json` also gives the maximum-weight QMI spanning tree, per-day tree edge occurrence, and the ratio of maximum spanning tree QMI weights to the nearest-neighbour chain QMI weights.

**Interpretation caveat:** quantum mutual information measures total correlation, not pairwise entanglement, direct couplings, or a minimal CNOT architecture. Summing QMI over pairs double-counts collective correlations. A maximum-weight QMI spanning tree is a heuristic candidate interaction topology, not an efficient preparation guarantee. In particular a weak pair may still take part in an essential multiparty entangling structure.
