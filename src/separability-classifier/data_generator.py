import qutip
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import json
from entanglement_generator import make_circuit
from params import num_qubits


def sign():
    return (-1) ** np.random.randint(2)


def make_matrix(n=2**num_qubits):
    matrix = np.array(
        sign() * np.random.random(n * n) + sign() * np.random.random(n * n) * 1j
    )
    matrix = matrix.reshape(n, n)
    return matrix


def validate(matrix):
    # Write validation for matrices.
    # Requirement is to check if this matrix is
    # a valid density matrix.

    # The following is not a good metric:
    # print(np.trace(matrix))
    # if abs(np.trace(matrix) - 1) > 1e-2:
    #     exit(0)
    return matrix


def embed(matrix):
    validate(matrix)
    flattened = (np.array(matrix).flatten()).tolist()
    embedding = []
    for i in flattened:
        embedding.extend([i.real, i.imag])
    return embedding


def get_separable(num: int, n=2**num_qubits, l=10):
    separable_dms = []
    for _ in range(num):
        dm_set = []
        for i in range(num_qubits):
            dm = np.array([validate(np.array(qutip.rand_dm(2))) for _ in range(l)])
            dm_set.append(dm)

        prob = np.array([random.random() for _ in range(l)])
        prob = prob / np.sum(prob)

        separable_dm = np.zeros((n, n))
        for i in range(l):
            tensor = dm_set[0][i]
            for j in range(1, num_qubits):
                tensor = np.tensordot(tensor, dm_set[j][i], axes=0)
            tensor = prob[i] * tensor
            separable_dm = np.add(separable_dm, tensor.reshape(n, n))
        separable_dms.append(embed(separable_dm))
    return separable_dms


def make_mixed_dm(pure_dm, n=2**num_qubits):
    if random.choice([False, True, True, True, True, True, True]):
        # Here, we shall apply random local unitaries on
        # the pure entangled state to produce mixed entangled states.
        u = [np.array(qutip.rand_unitary(2)) for i in range(num_qubits)]
        tensor = u[0]
        for i in range(1, num_qubits):
            tensor = np.tensordot(tensor, u[i], axes=0)
        entangled_dm = tensor.reshape(n, n) @ pure_dm
        return entangled_dm
    else:
        return pure_dm


def get_entangled(num: int, n=2**num_qubits):
    entangled_dms = []
    for _ in range(num):
        pure_entangled_dm = validate(make_circuit())
        mixed_entangled_dm = make_mixed_dm(pure_entangled_dm)
        entangled_dms.append(embed(mixed_entangled_dm))
    print("works")
    return entangled_dms
