from pyod.models.ocsvm import OCSVM
from pyod.models.so_gaal import SO_GAAL
from pyod.models.anogan import AnoGAN
from pyod.utils.example import visualize
from gensep import get_separable_states, get_random_states
import numpy as np
from config import TRAINCASES, TESTCASES


def accuracy(a, b):
    n = range(len(a))
    return sum([a[i] != b[i] for i in n])


def load_data(n: int, m: int):
    X_train = np.load(f"storage/test_data_{n}x{m}.npy")
    y_train = [0 for _ in range(TESTCASES)]
    return X_train, y_train


class Model:
    def __init__(self, n: int, m: int) -> None:
        self.X, self.y = load_data(n, m)

    def ocsvm(self):
        classifier = OCSVM(nu=0.9, gamma=0.5, kernel="rbf", verbose=1)
        classifier.fit(X=self.X, y=self.y)
        return classifier

    def gaal(self):
        classifier = SO_GAAL()
        classifier.fit(X=self.X, y=self.y)
        return classifier

    def anogan(self):
        classifier = AnoGAN()
        classifier.fit(X=self.X, y=self.y)
        return classifier


if __name__ == "__main__":
    model = Model(3, 3)
