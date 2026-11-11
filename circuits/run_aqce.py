
"""Run AQCE on one historical Heston quantum state."""

import numpy as np
from qulacs import QuantumState, QuantumCircuit
from heston.build_distributions import (
    read_calibrations,
    build_distributions,
)

from .build_distributions import (
    read_calibrations,
    build_distributions,
)
from .third_party.aqce.AQCE_from_python import AQCE_python


def main():
    # Begin with the first 2013 calibration only.
    calibrations = read_calibrations()[:1]

    # Construct the same eight-qubit Heston state
    # used in the covering-number experiments.
    batch = build_distributions(
        calibrations,
        n_qubits=8,
        cutoff=2.5,
        n_fft=16384,
        halfwidth=3.5,
        maturity=1.0,
        drift=0.05,
        verify_grid=True,
        fft_l1_tolerance=1e-4,
    )

    amplitudes = batch.amplitudes[0]

    # Convert the amplitude vector to a Qulacs state.
    target = QuantumState(8)
    target.load(
        np.asarray(amplitudes, dtype=complex)
    )

    # Generate a circuit automatically.
    # Start with a deliberately small gate budget.
    gates = AQCE_python(
        target_state=target,
        M_0=0,
        M_delta=1,
        M_max=3,
        N=1,
    )

    circuit = QuantumCircuit(8)
    for gate in gates:
        circuit.add_gate(gate)

    # Prepare the state using the generated circuit.
    prepared = QuantumState(8)
    circuit.update_quantum_state(prepared)

    # Check squared fidelity independently.
    fidelity = abs(
        np.vdot(amplitudes, prepared.get_vector())
    ) ** 2

    # Extract the interaction topology.
    topology = [
        gate.get_target_index_list()
        for gate in gates
    ]

    print("Date:", batch.dates[0])
    print("Two-qubit blocks:", len(gates))
    print("Squared fidelity:", fidelity)
    print("Interaction topology:", topology)
    print(circuit)


if __name__ == "__main__":
    main()
