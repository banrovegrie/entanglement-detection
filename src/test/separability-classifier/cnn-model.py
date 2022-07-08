import tensorflow as tf
import random
import numpy as np
from data_generator import get_entangled, get_separable
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from collections import Counter


def load_data():
    separable = get_separable(DATASET_SIZE)
    entangled = get_entangled(DATASET_SIZE)

    data = [[i, 1] for i in separable]
    data.extend([[i, 0] for i in entangled])

    return data


def process_data(data):
    return [[x, y] for x, y in data]


def please_plot(history):
    print(history.history)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy: CNN")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.savefig("./plots/result.png")


"""
Main function and global parameters
"""


def main():
    data = load_data()
    data = process_data(data)

    random.shuffle(data)
    split_ratio = 0.7
    split_size = int(split_ratio * len(data))

    train_data = data[:split_size]
    test_data = data[split_size:]

    train_X = np.array(list(map(lambda x: x[0], train_data)))
    train_Y = np.array(list(map(lambda x: x[1], train_data)))
    test_X = np.array(list(map(lambda x: x[0], test_data)))
    test_Y = np.array(list(map(lambda x: x[1], test_data)))

    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation="relu", input_shape=(8, 8, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_X, train_Y, epochs=EPOCHS, validation_data=(test_X, test_Y)
    )
    please_plot(history)
    print(Counter(test_Y))


if __name__ == "__main__":
    EPOCHS = 10
    DATASET_SIZE = 1000
    main()
