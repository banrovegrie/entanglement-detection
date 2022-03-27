import qutip
import numpy as np
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


def embed(matrix):
    flattened = (np.array(matrix).flatten()).tolist()
    embedding = []
    for i in flattened:
        embedding.extend([i.real, i.imag])
    return embedding


def get_separable(num: int, n=2**num_qubits):
    return []


def get_entangled(num: int, n=2**num_qubits):
    entangled_matrices = []
    for i in range(num):
        entangled_matrices.append(make_circuit())
    return entangled_matrices


print(get_entangled(100))
