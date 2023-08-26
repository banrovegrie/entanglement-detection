import qutip
import numpy as np
import random
from matplotlib import pyplot as plt
from cirq.linalg import is_unitary
from cirq.qis.states import validate_density_matrix


def embed(matrix: np.ndarray) -> list:
    x, y = matrix.shape

    # Validate the given density matrix
    try:
        validate_density_matrix(matrix, qid_shape=x)
    except:
        print(f"Not valid density matrix: {matrix}")
        exit(0)
    matrix = matrix.tolist()

    # Flatten and embed the density matrix by taking care that,
    # it has (n - 1) diagonal values and upper traingular matrix.
    flattened = []
    for i in range(x):
        for j in range(y):
            if i < j:
                flattened.append(matrix[i][j])

    # Embed the real and complex values respectively.
    embedding = []
    for i in flattened:
        embedding.extend([i.real, i.imag])

    # Embed diagonal matrices
    for i in range(1, x):
        embedding.append(matrix[i][i].real)

    return embedding


def get_separable_state(n: int, m: int, max_len: int = 50) -> np.ndarray:
    l = random.randint(1, max_len)
    prob = np.array([random.random() for _ in range(l)])
    prob = prob / np.sum(prob)

    separable_state = np.zeros((n * m, n * m))
    for i in range(l):
        tensor = np.kron(np.array(qutip.rand_dm(n)), np.array(qutip.rand_dm(m)))
        separable_state = np.add(separable_state, prob[i] * tensor)
    return separable_state


def get_separable_states(num: int, n: int, m: int) -> np.ndarray:
    separable_states = []
    for i in range(num):
        separable_states.append(embed(get_separable_state(n, m)))
        # print(i)
    return np.array(separable_states)


def get_random_states(num: int, n: int, m: int) -> np.ndarray:
    random_states = []
    for i in range(num):
        random_states.append(embed(np.array(qutip.rand_dm(n * m))))
        # print(i)
    return np.array(random_states)
