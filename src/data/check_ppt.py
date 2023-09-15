from tqdm import tqdm
import qutip as qt
import numpy as np


def is_PPT(random_state, n=2, m=2):
    """
    Check if a given state (in numpy array form) is PPT.
    """
    state = qt.Qobj(random_state, dims=[[n, n], [m, m]])
    partial_transpose = qt.partial_transpose(state, [0, 1])
    eigenvalues = partial_transpose.eigenenergies()

    if any(eig < 0 for eig in eigenvalues):
        return False
    return True


count = 0
# separable_states = []
random_states = np.load("random_states_2_2.npy")
for random_state in tqdm(random_states):
    if is_PPT(random_state):
        count += 1
        # separable_states.append(random_state)

print("The number of separable states are:", count)
# separable_states = np.array(separable_states)
# np.save('separable_states_2_2.npy', random_states)
