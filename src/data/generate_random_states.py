import qutip as qt
import numpy as np
from tqdm import tqdm

random_states = []


def generate_random_state(n=2, m=2):
    """
    This function generates random states of size $n\otimes m$.
    """
    state = qt.rand_dm(n*m, dims=[[n, n], [m, m]])
    return state.full()


for i in tqdm(range(100000)):
    random_state = generate_random_state()
    random_states.append(random_state)

random_states = np.array(random_states)
np.save('random_states_2_2.npy', random_states)