import random
import json
import numpy as np
import qiskit
import math
from params import num_qubits, depth, theta
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, transpile, Aer

# Simulate a given circuit.
def simulate(circuit):
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()
    output = result.get_statevector(circuit, decimals=3)
    return np.array(output)


def make_circuit():
    circuit = QuantumCircuit(num_qubits, num_qubits)

    # Ramdomly rotate the qubits to get randomized state.
    for i in range(num_qubits):
        circuit.r(random.uniform(0, 4 * math.pi), random.uniform(0, 4 * math.pi), i)

    # Strongly entangling ansatz state generator circuit.
    for d in range(depth):
        for i in range(num_qubits):
            circuit.u3(theta[d][i][0], theta[d][i][1], theta[d][i][2], i)
        for i in range(num_qubits):
            circuit.cnot(i, (i + 1) % num_qubits)

    # Measure the circuit.
    entangled_state = simulate(circuit)

    # Now, the entangled state obtained is a pure state.
    # First, we convert it into a density matrix and then,
    # we further apply local unitaries to get mixed entangled states.
    entangled_dm = np.outer(entangled_state, entangled_state.conj())
    return entangled_dm
