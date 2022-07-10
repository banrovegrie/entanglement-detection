# import libraries
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.utils.example import visualize
from gensep import get_separable_states, get_random_states
import numpy as np


def accuracy(a, b):
    n = range(len(a))
    return sum([a[i] != b[i] for i in n])


# X_train, X_test, y_train, y_test = generate_data(
#     n_train=5, n_test=1, n_features=3, behaviour="new"
# )


def load_data(n: int, m: int):
    X_train = np.load("storage/train_data.npy")
    y_train = [0 for _ in range(50000)]

    X_test = np.concatenate(
        (np.load("storage/test_data.npy")[:50], get_separable_states(50, 3, 3))
    )
    y_test = [0 for _ in range(100)]
    return X_train, X_test, y_train, y_test


def run(n: int, m: int) -> None:
    X_train, X_test, y_train, y_test = load_data(n, m)
    clf = OCSVM(nu=0.9, gamma=0.5, kernel="rbf", verbose=1)
    clf.fit(X_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print(y_test_pred)


run(3, 3)
