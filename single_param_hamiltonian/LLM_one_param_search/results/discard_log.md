# Discard log

- **Greedy forward block selection from the empty circuit (Stage 1, first attempt)**: abandoned after 2 steps.
  Fidelity is a *threshold* function of connectivity from the most-significant qubit: any set of pair
  blocks not forming a chain from qubit 1 stays at the product-state level (~0.66 on the 8-state subset),
  the chain prefix (1,2),(2,3),(3,4) reaches 0.868, through (4,5) 0.953 and the full 7-block chain 0.9977.
  Single-block greedy gains are therefore uninformative (first pick was (5,6) with 0.662 vs 0.644 for alternatives).
  Replaced by a structured search starting from chain prefixes (see stage1_topology.json).
- **Stage 1 v2 first run (results/stage1_topology_BUGGY_duplicate_labels.json)**: invalid. The fitter indexed
  coefficients by Pauli label, so repeated labels (block single-qubit Y terms, second-sweep generators) were frozen
  at zero. Families C/D/E of that run therefore under-estimate fidelity. Fixed by indexing coefficients by position; rerun.
