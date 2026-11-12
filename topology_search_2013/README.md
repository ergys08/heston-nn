# Shared Heston topology search and PQC

The 2013 search, topology search checkpoints, and final validated 8-block PQC are stored here.

From the **repository root**, regenerate the elementary-gate PQC (24 CX, 192 rotation angles) from its saved two-qubit unitaries:

```bash
python -m topology_search_2013.export_heston_pqc
```

The input is at `topology_search_2013/results/final/M08_7cce5211/unitaries.npz`.
The exported QASM, parameters, and numerical validation are at
`topology_search_2013/results/pqc_M08_7cce5211/`.
The full circuit diagram is `topology_search_2013/heston_pqc_circuit.png`.

To plot the current fixed PQC directly from the exporter without a QASM 3 importer:

```bash
python - <<'PY'
import numpy as np
from topology_search_2013.export_heston_pqc import make_template
with np.load("topology_search_2013/results/final/M08_7cce5211/unitaries.npz") as d:
    circuit, params = make_template(d["pairs"])
fig = circuit.draw(output="mpl", fold=40, scale=0.8)
fig.savefig("topology_search_2013/heston_pqc_circuit.png", dpi=250, bbox_inches="tight")
print(circuit.count_ops(), len(params))
PY
```
