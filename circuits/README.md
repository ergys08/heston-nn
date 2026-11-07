# Differentiable Heston PQCs

Three ansatze implemented in pure PyTorch, following the gate patterns and
rotation-block accounting of Koike-Akino (2026):

- `brickwork.py`: staggered even/odd nearest-neighbour CNOTs, with distinct
  local rotations immediately after **each** CNOT (on both its endpoints).
- `chain.py`: sequential CNOT(0,1), ..., CNOT(n-2,n-1), with a complete
  local rotation bank between entangling sweeps.
- `ring.py`: the same sequential chain plus CNOT(n-1,0), with complete
  local rotation banks between sweeps.

All circuits start at |0...0>, use qubit 0 as the least significant bit,
and return complex128 normalized amplitudes. The only required library is
PyTorch, not Qiskit. Rotation parameters preserve autograd and may be batched.

The default local block is `Ry(alpha) @ Rz(beta)`, with the rightmost
matrix applied first; the optional `rotation="rot"` block is
`Rz(alpha) @ Ry(beta) @ Rz(gamma)`.

## Usage

```python
import torch
from circuits import (
    run_brickwork_circuit, brickwork_num_parameters,
    run_chain_circuit, chain_num_parameters,
    run_ring_circuit, ring_num_parameters,
)

n, layers = 8, 18
p = brickwork_num_parameters(n, layers)
angles = torch.nn.Parameter(torch.zeros(p, dtype=torch.float64))
state = run_brickwork_circuit(n, layers, angles)
loss = 1 - torch.abs(torch.vdot(target_state, state)) ** 2
loss.backward()
```

Input: `parameters` of shape `(P,)` or `(batch, P)`, where
`P = *_num_parameters(n, layers, rotation)`. For an NN producing a batch
of angles, the output is `(batch, 2**n)`; for a vector it is `(2**n,)`.

## Eight-qubit heuristic depths

| Topology | Layers | CNOTs | Local rotation blocks | Angles (RyRz) |
|---|---:|---:|---:|---:|
| Brickwork | 18 | 126 | 260 | 520 |
| Chain | 18 | 126 | 152 | 304 |
| Ring | 16 | 128 | 136 | 272 |

These depths are motivated by the 124-CNOT generic **complex** eight-qubit
state lower bound. They are heuristic, not a requirement for Heston states,
nor a guarantee of variational expressivity or successful optimization.

## Validation

Run `python -m pytest -q tests/test_circuits.py` at repository root.
Tests check gradients, batch shapes, norms, zero-angle identity action,
CNOT bit ordering, gate counts, and rotation matrix convention.
