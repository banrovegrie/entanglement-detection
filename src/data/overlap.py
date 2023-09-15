import numpy as np
from tqdm import tqdm

random_states = np.load("random_states_2_2.npy")
separable_states = np.load("separable_states_2_2.npy")

count = 0
for separable_state in tqdm(separable_states):
    if separable_state in random_states:
        count += 1

print("Number of common states:", count)