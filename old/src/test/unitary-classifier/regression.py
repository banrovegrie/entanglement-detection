"""
Logistic Regression classifier for unitary matrices.
"""
from model import load_data, process_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random


def main():
    data = load_data()
    random.shuffle(data)
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=0
    )

    logisticRegr = LogisticRegression(max_iter=100000)
    logisticRegr.fit(x_train, y_train)
    # predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)
    print(score)


if __name__ == "__main__":
    main()
