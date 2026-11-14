"""Real-valued generators for odd-Y Pauli strings on eight qubits.

Convention (matches single_param_hamiltonian/): paper qubit 1 is the most
significant bit of the 256-dimensional basis index.  A Pauli string is written
as an 8-character label, e.g. "IIYZIIII" means Y on qubit 3 and Z on qubit 4.
The compact label used in the earlier fits ("Y3Z4") is also accepted.

Only strings with an ODD number of Y factors are used: for these, P is purely
imaginary, hence A = -iP is a real antisymmetric matrix and exp(theta*A) is a
real orthogonal matrix.  Because the Heston targets are real nonnegative
vectors this restriction costs nothing and keeps all arithmetic real.

For a Pauli string P with X/Y flip mask f and Z/Y sign mask z,
    (A v)[y] = sgn[y] * v[y ^ f],    sgn[y] = (-1)^((n_Y-1)/2) * (-1)^{popcount((y^f) & z)}.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

N_QUBITS = 8
DIM = 1 << N_QUBITS
_COMPACT = re.compile(r"([XYZ])(\d)")


def normalise_label(label: str) -> str:
    """Accept 'IIYZIIII' or 'Y3Z4' and return the 8-character form."""
    label = label.strip()
    if len(label) == N_QUBITS and set(label) <= set("IXYZ"):
        return label
    pieces = _COMPACT.findall(label)
    if not pieces or "".join(a + b for a, b in pieces) != label:
        raise ValueError(f"Bad Pauli label {label!r}")
    chars = ["I"] * N_QUBITS
    for axis, q in pieces:
        q = int(q) - 1
        if not 0 <= q < N_QUBITS or chars[q] != "I":
            raise ValueError(f"Bad Pauli label {label!r}")
        chars[q] = axis
    return "".join(chars)


def compact_label(label: str) -> str:
    """'IIYZIIII' -> 'Y3Z4'."""
    label = normalise_label(label)
    return "".join(f"{a}{q + 1}" for q, a in enumerate(label) if a != "I")


def support(label: str) -> tuple[int, ...]:
    """Zero-based paper qubits (0 = MSB) on which the string acts."""
    label = normalise_label(label)
    return tuple(q for q, a in enumerate(label) if a != "I")


def weight(label: str) -> int:
    return len(support(label))


@dataclass(frozen=True)
class Generator:
    label: str
    perm: np.ndarray   # index permutation y -> y ^ f
    sign: np.ndarray   # real signs so that (A v) = sign * v[perm]

    def apply(self, v: np.ndarray) -> np.ndarray:
        """A v for v of shape (..., 256)."""
        return self.sign * v[..., self.perm]


@lru_cache(maxsize=None)
def generator(label: str) -> Generator:
    label = normalise_label(label)
    n_y = label.count("Y")
    if n_y % 2 == 0:
        raise ValueError(f"{label}: need an odd number of Y factors for a real generator")
    f = z = 0
    for q, a in enumerate(label):
        bit = 1 << (N_QUBITS - 1 - q)
        if a in "XY":
            f |= bit
        if a in "YZ":
            z |= bit
    y = np.arange(DIM)
    perm = y ^ f
    parity = np.array([bin(int(v)).count("1") & 1 for v in (perm & z)])
    sign = np.where(parity == 0, 1.0, -1.0) * ((-1.0) ** ((n_y - 1) // 2))
    return Generator(label, perm.astype(np.intp), sign.astype(np.float64))


def dense_generator(label: str) -> np.ndarray:
    """Dense real matrix A = -iP (for verification only)."""
    g = generator(label)
    A = np.zeros((DIM, DIM))
    A[np.arange(DIM), g.perm] = g.sign
    return A


def odd_y_strings(qubits: tuple[int, ...]) -> list[str]:
    """All odd-Y strings whose support is exactly `qubits` (0-based)."""
    out = []
    for axes in itertools.product("XYZ", repeat=len(qubits)):
        if axes.count("Y") % 2 == 1:
            chars = ["I"] * N_QUBITS
            for q, a in zip(qubits, axes):
                chars[q] = a
            out.append("".join(chars))
    return out


def all_strings_up_to_weight(max_weight: int, *, max_span: int | None = None) -> list[str]:
    """Odd-Y strings of weight 1..max_weight; optionally restrict qubit span."""
    out = []
    for w in range(1, max_weight + 1):
        for qubits in itertools.combinations(range(N_QUBITS), w):
            if max_span is not None and qubits[-1] - qubits[0] > max_span:
                continue
            out.extend(odd_y_strings(qubits))
    return out


def pair_block_strings(i: int, j: int) -> list[str]:
    """The six so(4) generators on qubits (i, j): Y_i, Y_j, Y_iX_j, Y_iZ_j, X_iY_j, Z_iY_j."""
    assert i != j
    out = odd_y_strings((i,)) + odd_y_strings((j,))
    out += odd_y_strings(tuple(sorted((i, j))))
    return out


if __name__ == "__main__":
    # Self-test against explicit Kronecker products.
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    P = {"I": I2, "X": X, "Y": Y, "Z": Z}
    rng = np.random.default_rng(0)
    for label in ["YIIIIIII", "IIYZIIII", "IIZYIIII", "IYXIIIII", "YYYIIIII", "IIYZZIII", "IYIIIIXI", "IIIIIIIY"]:
        M = np.array([[1.0]])
        for a in label:
            M = np.kron(M, P[a])
        A = -1j * M
        assert np.allclose(A.imag, 0)
        assert np.allclose(dense_generator(label), A.real), label
        v = rng.normal(size=DIM)
        assert np.allclose(generator(label).apply(v), A.real @ v)
        assert compact_label(label) == compact_label(normalise_label(compact_label(label)))
    assert normalise_label("Y4Z3") == "IIZYIIII"
    assert len(all_strings_up_to_weight(2)) == 8 + 28 * 4
    print("paulis.py self-test passed")
