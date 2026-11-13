"""Parallel-sequential (PS) architectures using the Heston PQC's exact two-qubit gate template.

Wei & Malz, Quantum 10, 2079 (2026), https://doi.org/10.22331/q-2026-04-21-2079

Parameters:
    n_qubits: chain length (>=3)
    M: repeated gates in each nearest-neighbour stack (>=1)
    l: length of a sequential chunk (2 <= l <= n_qubits-1)
    q: overlap (1 <= q <= l); q=1 gives no repeated boundary bonds

Chunks start at bonds 0,l,2*l,... and contain l+q-1 successive
nearest-neighbour bonds, clipped to the last bond. The stacks in each
chunk are staggered by two clock slots. Overlaps are scheduled to avoid
qubit conflicts, and the actual disjoint layers are returned.

q=1, l=2 -> depth-2M brickwall circuit.
q=1, l=n_qubits-1 -> M-layer sequential staircase.

For a block on (a,b) with a<b, the template is identical to
topology_search_2013.export_heston_pqc.make_template:
    four RZ(lambda)-RY(theta)-RZ(phi) rotation slots on each wire,
    interleaved with exactly three CX(b -> a) gates.

Per-block angles are ordered [slot, wire, (theta, phi, lambda)]
with local wire 0 = global qubit b and local wire 1 = global qubit a.
The ordered pair list is compatible with circuits.fixed_topology_torch.

This creates a parameterised layout; it does not optimise its angles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

Pair = tuple[int, int]


@dataclass(frozen=True)
class PSLayout:
    n_qubits: int
    M: int
    l: int
    q: int
    layers: tuple[tuple[Pair, ...], ...]

    @property
    def pairs(self) -> tuple[Pair, ...]:
        return tuple(p for layer in self.layers for p in layer)

    @property
    def n_blocks(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def two_qubit_depth(self) -> int:
        return len(self.layers)

    @property
    def n_cx(self) -> int:
        return 3 * self.n_blocks

    @property
    def n_rz(self) -> int:
        return 16 * self.n_blocks

    @property
    def n_ry(self) -> int:
        return 8 * self.n_blocks

    @property
    def n_angles(self) -> int:
        return 24 * self.n_blocks


def build_ps_layout(n_qubits: int, M: int, l: int, q: int) -> PSLayout:
    """Schedule a finite one-dimensional PS architecture as disjoint layers.

    The nominal staggered offset of a gate is local_bond_offset + 2*stack.
    At chunk overlaps, conflicting gates are postponed deterministically.
    Thus reported depth describes the actual schedule rather than assuming
    the asymptotic depth expression from the reference paper.
    """
    if any(type(x) is not int for x in (n_qubits, M, l, q)):
        raise TypeError("n_qubits, M, l and q must be integers")
    if n_qubits < 3:
        raise ValueError("n_qubits must be >= 3")
    if M < 1:
        raise ValueError("M must be >= 1")
    if not 2 <= l <= n_qubits - 1:
        raise ValueError("require 2 <= l <= n_qubits-1")
    if not 1 <= q <= l:
        raise ValueError("require 1 <= q <= l")

    # Entries are (preferred clock, chunk, offset, stack, pair).
    events = []
    for chunk, start in enumerate(range(0, n_qubits - 1, l)):
        extent = min(l + q - 1, n_qubits - 1 - start)
        for stack in range(M):
            for offset in range(extent):
                a = start + offset
                events.append((offset + 2 * stack, chunk, offset, stack, (a, a + 1)))
    events.sort()

    scheduled: list[list[Pair]] = []
    busy: list[set[int]] = []
    last_used = [-1] * n_qubits
    for desired, _chunk, _offset, _stack, pair in events:
        a, b = pair
        clock = max(desired, last_used[a] + 1, last_used[b] + 1)
        while clock < len(busy) and (a in busy[clock] or b in busy[clock]):
            clock += 1
        while len(scheduled) <= clock:
            scheduled.append([])
            busy.append(set())
        scheduled[clock].append(pair)
        busy[clock].update((a, b))
        last_used[a] = last_used[b] = clock

    return PSLayout(n_qubits, M, l, q,
                    tuple(tuple(layer) for layer in scheduled if layer))


def ps_topology(n_qubits: int, M: int, l: int, q: int) -> list[Pair]:
    """Ordered nearest-neighbour pairs, usable by the existing SVD optimiser."""
    return list(build_ps_layout(n_qubits, M, l, q).pairs)


def build_ps_circuit(
    n_qubits: int, M: int, l: int, q: int, *,
    angles: Any | None = None,
    barriers: bool = False,
):
    """Return (QuantumCircuit, parameter_list) using the exact 3-CX block.

    angles can be a flat vector of length 24*n_blocks or have shape
    (n_blocks,4,2,3). With angles=None the template remains symbolic.
    """
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
    except ImportError as exc:
        raise ImportError("Install Qiskit: pip install 'qiskit>=2,<3'") from exc

    layout = build_ps_layout(n_qubits, M, l, q)
    values = None
    if angles is not None:
        values = np.asarray(angles, dtype=float)
        allowed = ((layout.n_angles,), (layout.n_blocks, 4, 2, 3))
        if values.shape not in allowed or not np.isfinite(values).all():
            raise ValueError(f"Expected finite angles of shape {allowed}")
        values = values.reshape(layout.n_blocks, 4, 2, 3)

    circuit = QuantumCircuit(n_qubits, name=f"PS_N{n_qubits}_M{M}_l{l}_q{q}")
    parameters = []
    block_id = 0
    for layer_idx, pairs in enumerate(layout.layers):
        for a, b in pairs:
            for slot in range(4):
                for wire, qubit in enumerate((b, a)):
                    tag = f"g{block_id+1:03d}_s{slot}_w{wire}"
                    theta = Parameter(f"{tag}_theta")
                    phi = Parameter(f"{tag}_phi")
                    lam = Parameter(f"{tag}_lambda")
                    parameters.extend((theta, phi, lam))
                    th, ph, la = ((theta, phi, lam) if values is None
                                  else values[block_id, slot, wire])
                    circuit.rz(la, qubit)
                    circuit.ry(th, qubit)
                    circuit.rz(ph, qubit)
                if slot < 3:
                    circuit.cx(b, a)
            block_id += 1
        if barriers and layer_idx < len(layout.layers) - 1:
            circuit.barrier()

    assert len(parameters) == layout.n_angles
    assert circuit.count_ops().get("cx", 0) == layout.n_cx
    return circuit, parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--qasm", type=Path, help="Optional OpenQASM 3 output")
    parser.add_argument("--diagram", type=Path, help="Optional text diagram output")
    parser.add_argument("--barriers", action="store_true")
    args = parser.parse_args()

    layout = build_ps_layout(args.qubits, args.M, args.l, args.q)
    print(f"PS N={layout.n_qubits}, M={layout.M}, l={layout.l}, q={layout.q}")
    print(f"Blocks={layout.n_blocks}, two-qubit depth={layout.two_qubit_depth}")
    print(f"CX={layout.n_cx}, RZ={layout.n_rz}, RY={layout.n_ry}, angles={layout.n_angles}")
    print("Ordered pairs:", list(layout.pairs))
    for t, layer in enumerate(layout.layers):
        print(f"Layer {t}: {list(layer)}")

    if args.qasm or args.diagram:
        circuit, _ = build_ps_circuit(args.qubits, args.M, args.l, args.q,
                                      barriers=args.barriers)
        if args.qasm:
            from qiskit import qasm3
            args.qasm.parent.mkdir(parents=True, exist_ok=True)
            args.qasm.write_text(qasm3.dumps(circuit))
            print("Wrote", args.qasm)
        if args.diagram:
            args.diagram.parent.mkdir(parents=True, exist_ok=True)
            args.diagram.write_text(str(circuit.draw(output="text", fold=140)))
            print("Wrote", args.diagram)


if __name__ == "__main__":
    main()
