from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from cirq.linalg import is_unitary


def embed(matrix):
    flattened = (np.array(matrix).flatten()).tolist()
    embedding = []
    for i in flattened:
        embedding.extend([i.real, i.imag])
    return embedding


def get_unitaries(num: int, n: int) -> list:
    unitary_data = []
    for _ in range(num):
        u = np.array(rand_unitary(n))
        unitary_data.append(embed(u))
    return unitary_data


print(get_unitaries(3, 3))