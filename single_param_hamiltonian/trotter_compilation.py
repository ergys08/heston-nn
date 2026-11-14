#!/usr/bin/env python3
"""Compile the fitted 2013 Heston Hamiltonian into Rx/Ry/Rz/CNOT circuits.

Uses results/fitted_hamiltonian.json from hamiltonian_fitting.py, *not*
hard-coded coefficients. Does not require Qiskit. Produces a parameterised
OpenQASM 2.0 custom gate with ONE scalar parameter tau and an example call.

Examples (from any working directory):
    python single_param_hamiltonian/trotter_compilation.py --tau 1 --verify
    python single_param_hamiltonian/trotter_compilation.py --tau 0.5 --order 2 --steps 1 --verify
    python single_param_hamiltonian/trotter_compilation.py --tau -0.2 --steps 3 --out my_circuit.qasm

Qubit notation: paper qubit 1 = OpenQASM q[0] = most-significant bit.
The circuit implements a Trotter approximation to U(tau)=exp(-i*tau*H0),
NOT state preparation from |00000000>. To transport the Jan 7 state, for
example, supply that state as the input to the circuit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_HAMILTONIAN = HERE / "results" / "fitted_hamiltonian.json"
DEFAULT_STATES = HERE / "results" / "five_initial_state_tangents.npz"
N_QUBITS = 8
_LABEL_PATTERN = re.compile(r"([XYZ])(\d+)")


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]  # 0-based, matching OpenQASM
    constant: float = 0.0   # angle = constant + tau * scale
    scale: float = 0.0

    def angle(self, tau: float) -> float:
        return self.constant + tau * self.scale


def parse_pauli(label: str, n_qubits: int = N_QUBITS) -> dict[int, str]:
    """Return a zero-based {qubit: 'X'|'Y'|'Z'} mapping."""
    pieces = _LABEL_PATTERN.findall(label)
    if not pieces or "".join(f"{axis}{index}" for axis,index in pieces) != label:
        raise ValueError(f"Invalid Pauli label: {label!r}")
    if len(pieces) > 2:
        raise ValueError(f"Only one- and two-qubit terms supported: {label}")
    pauli = {}
    for axis, index in pieces:
        q = int(index) - 1
        if not 0 <= q < n_qubits or q in pauli:
            raise ValueError(f"Invalid or repeated qubit in {label!r}")
        pauli[q] = axis
    if list(pauli.values()).count("Y") != 1:
        raise ValueError(f"Expected exactly one Y in {label!r}")
    return pauli


def load_hamiltonian(path: Path = DEFAULT_HAMILTONIAN):
    """Return ordered list of (Pauli label, real coefficient) from fit JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    coeffs = data["coefficients"]
    support = data.get("support", list(coeffs))
    if len(support) != len(set(support)) or set(support) != set(coeffs):
        raise ValueError("Support and coefficient keys do not agree")
    terms = []
    for name in support:
        parse_pauli(name)
        value = float(coeffs[name])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite coefficient for {name}")
        terms.append((name, value))
    if not terms:
        raise ValueError("Hamiltonian has no terms")
    return terms


def trotter_schedule(n_terms: int, steps: int, order: int):
    """Produce ordered (term_index, fraction_of_tau) rotations.

    1st-order: [P1 ... Pn]^r; 2nd-order: symmetric Strang [P1/2 ...
    P(n-1)/2, Pn, P(n-1)/2 ... P1/2]^r. Adjacent repetitions of the
    same Pauli rotation are merged exactly.
    """
    if steps < 1 or order not in (1, 2) or n_terms < 1:
        raise ValueError("Require steps >= 1, order in {1,2}, terms >= 1")
    if order == 1:
        one_step = [(i, 1.0 / steps) for i in range(n_terms)]
    else:
        half = 0.5 / steps
        one_step = (
            [(i, half) for i in range(n_terms-1)]
            + [(n_terms-1, 1.0 / steps)]
            + [(i, half) for i in range(n_terms-2, -1, -1)]
        )
    combined = []
    for index, amount in one_step * steps:
        if combined and combined[-1][0] == index:
            previous, prev_amount = combined[-1]
            combined[-1] = (previous, prev_amount + amount)
        else:
            combined.append((index, amount))
    return combined


def pauli_rotation(label: str, angle_scale: float):
    """Gates for exp[-i (tau * angle_scale/2) P].

    B_Y=Rx(-pi/2), B_X=Ry(pi/2), B_Z=I with B Z B^dag = P.
    The execution order is B^dag, CX, Rz(angle), CX, B.
    """
    pauli = parse_pauli(label)
    if len(pauli) == 1:
        q = next(iter(pauli))
        return [Gate("ry", (q,), scale=angle_scale)]
    y = next(q for q, a in pauli.items() if a == "Y")
    other = next(q for q in pauli if q != y)
    other_axis = pauli[other]
    gates = [Gate("rx", (y,), constant=math.pi / 2)]
    if other_axis == "X":
        gates.append(Gate("ry", (other,), constant=-math.pi / 2))
    gates.extend([
        Gate("cx", (other, y)),
        Gate("rz", (y,), scale=angle_scale),
        Gate("cx", (other, y)),
    ])
    if other_axis == "X":
        gates.append(Gate("ry", (other,), constant=math.pi / 2))
    gates.append(Gate("rx", (y,), constant=-math.pi / 2))
    return gates


def compile_trotter(terms, *, order: int = 1, steps: int = 1):
    """Return fixed gate template; every angle is constant + scale * tau."""
    sequence = []
    for index, weight in trotter_schedule(len(terms), steps, order):
        label, coefficient = terms[index]
        sequence.extend(pauli_rotation(label, 2.0 * weight * coefficient))
    return sequence


def gate_counts(gates):
    return dict(sorted(Counter(g.name for g in gates).items()))


def _angle_expr(gate: Gate):
    """Angle in QASM syntax, inside a gate(tau) definition."""
    if gate.scale == 0.0:
        if math.isclose(gate.constant, math.pi/2):
            return "pi/2"
        if math.isclose(gate.constant, -math.pi/2):
            return "-pi/2"
        return f"{gate.constant:.17g}"
    if gate.constant == 0:
        return f"({gate.scale:.17g})*tau"
    return f"({gate.constant:.17g}+({gate.scale:.17g})*tau)"


def openqasm2(gates, *, tau: float, n_qubits: int = N_QUBITS,
               order: int = 1, steps: int = 1):
    """Parameterised custom gate and a numeric invocation at chosen tau."""
    if not math.isfinite(tau):
        raise ValueError("tau must be finite")
    formal_qubits = ",".join(f"q{i}" for i in range(n_qubits))
    actual_qubits = ",".join(f"q[{i}]" for i in range(n_qubits))
    out = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "// Paper qubit 1 = q[0] (most significant); paper qubit 8 = q[7].",
        f"// Suzuki-Trotter order {order}, {steps} step(s).",
        "// H0 read from fitted_hamiltonian.json; one independent parameter tau.",
        f"gate heston_trotter(tau) {formal_qubits} {{",
    ]
    for g in gates:
        if g.name == "cx":
            a, b = g.qubits
            out.append(f"  cx q{a},q{b};")
        else:
            out.append(f"  {g.name}({_angle_expr(g)}) q{g.qubits[0]};")
    out.extend([
        "}",
        f"qreg q[{n_qubits}];",
        "// IMPORTANT: q starts in |0...0>; supply/prepare a Heston input state",
        "// before the heston_trotter call when testing state transport.",
        f"heston_trotter({tau:.17g}) {actual_qubits};",
        "",
    ])
    return "\n".join(out)


def apply_gates(gates, state, tau: float):
    """Numerically evolve an arbitrary n-qubit state with the compiled gates."""
    psi = np.asarray(state, dtype=complex).copy()
    n = int(round(math.log2(psi.size)))
    if 2**n != psi.size:
        raise ValueError("State length must be a power of two")
    psi = psi.reshape((2,) * n)
    for gate in gates:
        if gate.name == "cx":
            c, t = gate.qubits
            if c == t or not (0 <= c < n and 0 <= t < n):
                raise ValueError(f"Invalid CNOT: {gate}")
            # Map tensor axes into [control, target, all remaining].
            axes = [c, t] + [i for i in range(n) if i not in (c,t)]
            arr = np.transpose(psi, axes).reshape(2, 2, -1)
            arr[1] = arr[1, ::-1, :]
            psi = np.transpose(arr.reshape((2,) * n), np.argsort(axes))
        else:
            q = gate.qubits[0]
            a = gate.angle(tau)
            cs, sn = np.cos(a/2), np.sin(a/2)
            if gate.name == "rx":
                mat = np.array([[cs, -1j*sn],[-1j*sn,cs]])
            elif gate.name == "ry":
                mat = np.array([[cs, -sn],[sn,cs]])
            elif gate.name == "rz":
                mat = np.diag([np.exp(-.5j*a),np.exp(.5j*a)])
            else:
                raise ValueError(f"Unsupported gate {gate.name}")
            psi = np.moveaxis(np.tensordot(mat, psi, axes=([1],[q])), 0, q)
    return psi.reshape(-1)


def exact_state(terms, initial, tau):
    """Reference evolution using exact sparse expm_multiply, not Trotter."""
    from scipy.sparse import csr_matrix, kron
    from scipy.sparse.linalg import expm_multiply

    paulis = {
        "I": csr_matrix(np.eye(2, dtype=complex)),
        "X": csr_matrix([[0,1],[1,0]],dtype=complex),
        "Y": csr_matrix([[0,-1j],[1j,0]],dtype=complex),
        "Z": csr_matrix([[1,0],[0,-1]],dtype=complex),
    }
    dim = 2**N_QUBITS
    H = csr_matrix((dim,dim),dtype=complex)
    for label, coeff in terms:
        axes = parse_pauli(label)
        product = csr_matrix([[1]],dtype=complex)
        for q in range(N_QUBITS):
            product = kron(product, paulis[axes.get(q, "I")], format="csr")
        H = H + coeff * product
    return expm_multiply(-1j*tau*H, np.asarray(initial,dtype=complex))


def verify_circuit(gates, terms, tau, states_path=DEFAULT_STATES):
    """Compare compiled circuit with exact exp(-i*tau*H0) on Jan 7 input."""
    if Path(states_path).exists():
        with np.load(states_path) as data:
            initial = data["states"][3]
            target = data["states"][4]
        label = "2013-01-07 (Jan 8 target)"
    else:
        # Fall back to a deterministic, generic state if the NPZ is absent.
        rng = np.random.default_rng(613)
        initial = rng.normal(size=256)+1j*rng.normal(size=256)
        initial = initial / np.linalg.norm(initial)
        target = None
        label = "deterministic random test state"
    output = apply_gates(gates, initial, tau)
    exact = exact_state(terms, initial, tau)
    circuit_fidelity = abs(np.vdot(exact,output))**2
    print("Verification input:", label)
    print(f"Fidelity(circuit, exact Hamiltonian) = {circuit_fidelity:.12f}")
    print(f"State-vector 2-norm error = {np.linalg.norm(output-exact):.9g}")
    if target is not None:
        print(f"Target fidelity, Trotter circuit = {abs(np.vdot(target,output))**2:.9f}")
        print(f"Target fidelity, exact Hamiltonian = {abs(np.vdot(target,exact))**2:.9f}")
    if not np.isclose(np.linalg.norm(output), 1, atol=1e-10):
        raise AssertionError("Circuit failed norm-preservation check")
    return float(circuit_fidelity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian",type=Path,default=DEFAULT_HAMILTONIAN,
                        help="Output JSON of hamiltonian_fitting.py")
    parser.add_argument("--tau",type=float,default=1.0,
                        help="Evolution scalar; only variable circuit parameter")
    parser.add_argument("--order",type=int,choices=[1,2],default=1,
                        help="Trotter order: 1 = Lie, 2 = symmetric Suzuki")
    parser.add_argument("--steps",type=int,default=1,
                        help="Number of repeated Trotter steps")
    parser.add_argument("--out",type=Path,default=None,
                        help="Output OpenQASM 2.0 file (default: results/)")
    parser.add_argument("--verify",action="store_true",
                        help="Simulate compiled circuit and exact Hamiltonian")
    parser.add_argument("--states",type=Path,default=DEFAULT_STATES,
                        help="Optional NPZ containing initial/target Heston states")
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.tau):
        parser.error("--steps must be >=1 and --tau must be finite")
    terms = load_hamiltonian(args.hamiltonian)
    gates = compile_trotter(terms,steps=args.steps,order=args.order)
    output = args.out or (HERE/"results"/f"trotter_order{args.order}_steps{args.steps}.qasm")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(openqasm2(gates,tau=args.tau,order=args.order,steps=args.steps),encoding="utf-8")
    counts = gate_counts(gates)
    print(f"Hamiltonian: {len(terms)} Pauli terms loaded from {args.hamiltonian}")
    print(f"U(tau)=exp(-i*tau*H0), tau={args.tau:g}, order={args.order}, steps={args.steps}")
    print("Gate counts:",counts)
    print("Total CNOTs:",counts.get("cx",0))
    print("QASM file:",output)
    if args.verify:
        verify_circuit(gates,terms,args.tau,states_path=args.states)


if __name__ == "__main__":
    main()
