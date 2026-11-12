# Iterative MPS state preparation

Reference: J. Iaconis, S. Johri and E. Y. Zhu, *Quantum state preparation of normal distributions using matrix product states*, npj Quantum Information **10**, 15 (2024), https://doi.org/10.1038/s41534-024-00805-0.

This implementation accepts an input statevector (such as one of the 256-amplitude historical Heston targets) and implements iterative bond-2 MPS compression and circuit extraction. It does **not** implement the paper's Irwin–Hall data-loading method or its large-system MPS contraction pipeline.

- `mps_decomposition.py`: right-to-left SVD decomposition, configurable bond truncation, reconstructed states, discarded weights.
- `circuit.py`: a 7-nearest-neighbor-two-qubit-block layer for eight qubits, iterative adjoint-residual method, NumPy simulator, optional Qiskit circuit export, and benchmark CLI.
- `tests/test_mps.py`: mathematical reconstruction, qubit-ordering, iterative circuit, and optional Qiskit tests.

From the repository root:

```bash
python -m pytest tests/test_mps.py -q
python -m MPS.circuit --layers 1 2 3 --max-states 10 --output MPS/pilot_2013.csv
python -m MPS.circuit --layers 1 2 3 4 --output MPS/mps_fidelities.csv
python -m MPS.circuit --layers 1 --max-states 1 --qasm --output MPS/first_state.csv
```

The last command requires Qiskit installed and exports compiled QASM 2, checking the compiled statevector against the precompiled numerical result.

## Python API

```python
import numpy as np
from MPS.mps_decomposition import decompose_state
from MPS.circuit import prepare_iterative_mps

with np.load("aqce_results_2013/target_states.npz") as data:
    target = data["amplitudes"][0]

mps = decompose_state(target, max_bond=2)
print(mps.bond_dimensions, mps.fidelity)
result = prepare_iterative_mps(target, n_layers=3)
print(result.fidelities)
print(result.n_two_qubit_gates, result.nominal_real_cx_count)
print(abs(np.vdot(target, result.prepare_statevector()))**2)
qc = result.to_qiskit(decompose_to_cx=True)   # Qiskit optional
print(qc.count_ops())  # Actual compiled gate counts
```

For eight qubits, D iterative layers have 7D abstract two-qubit gates; the reference's **nominal** two-CNOT real SO(4) synthesis gives 14D CX, i.e. 14, 28, and 42 for D=1, 2, and 3. In contrast, the previously validated shared topology uses 24 CX. Compare fidelity at matched targets and actual compiled CNOT counts to assess relative resources.

**Limitations:** The implementation uses dense statevectors (2**n complex amplitudes) to evolve the residual, not the paper's large-n native-MPS representation. The greedy iterative scheme is not globally optimised. One isometry's arbitrary completion can influence later residuals. The reported nominal CX count assumes real SO(4) gates; check concrete Qiskit synthesis and device routing separately.
