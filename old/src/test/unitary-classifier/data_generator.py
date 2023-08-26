from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from cirq.linalg import is_unitary
import torch


def sign():
    return (-1) ** np.random.randint(2)


def make_matrix(n):
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


def get_unitaries(num: int, n: int) -> list:
    unitary_data = []
    for _ in range(num):
        u = np.array(rand_unitary(n))
        unitary_data.append(embed(u))
    return unitary_data


def get_non_unitaries(num: int, n) -> list:
    non_unitary_data = []

    for _ in range(num):
        nu = make_matrix(n)
        if is_unitary(nu):
            num += 1
            continue
        non_unitary_data.append(embed(nu))
    return non_unitary_data


# print(get_unitaries(10, 3))
# print(get_non_unitaries(10, 3))
# with open("dump.txt", "w") as f:
#     unitaries = get_unitaries(10000, 3)
#     non_unitaries = get_non_unitaries(10000, 3)

#     data = [(unitary, torch.tensor(1)) for unitary in unitaries]
#     data.extend([(non_unitary, torch.tensor(0)) for non_unitary in non_unitaries])
