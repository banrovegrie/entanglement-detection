from data_generator import get_entangled, get_separable
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter
from os import path
import json


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 150 --> 150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()

        # Linear function 3: 150 --> 150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.elu3 = nn.ELU()

        # Linear function 4 (readout): 150 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return F.log_softmax(out, dim=0)


def train(model, optimizer, criterion, data):
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in tqdm(data):
            optimizer.zero_grad()

            prediction = model.forward(x)
            loss = criterion(prediction, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print(f"epoch {epoch+1}: {epoch_loss/len(data)}")


def test(model, data):
    model.eval()
    count = 0
    for x, y in data:
        prediction = model.forward(x)
        prediction = torch.argmax(prediction)
        if prediction == y:
            count += 1

    print("accuracy", count / len(data) * 100)


def get_stats(data):
    return Counter(map(lambda x: int(x[1]), data))


def load_data():
    separable = get_separable(DATASET_SIZE)
    entangled = get_entangled(DATASET_SIZE)

    data = [[i, 1] for i in separable]
    data.extend([[i, 0] for i in entangled])

    return data


def process_data(data):
    return [(torch.tensor(x), torch.tensor(y)) for x, y in data]


"""
Main function and global parameters
"""

EPOCHS = 10
DATASET_SIZE = 1000
# MATRIX_SIZE = 5


def main():
    data = load_data()
    data = process_data(data)

    random.shuffle(data)
    split_ratio = 0.7
    split_size = int(split_ratio * len(data))

    train_data = data[:split_size]
    test_data = data[split_size:]

    model = ANNModel(data[0][0].shape[0], 8, 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()
    train(model, optimizer, criterion, train_data)
    test(model, test_data)

    print("Test data y distribution", get_stats(test_data))


if __name__ == "__main__":
    main()
