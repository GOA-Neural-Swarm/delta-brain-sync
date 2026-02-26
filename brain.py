import numpy as np
from scipy import optimize

class Brain:
    def __init__(self):
        self.weights = np.random.rand()
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        learning_rate = 0.01
        for _ in range(1000):
            outputs = self.forward_pass(inputs)
            error = targets - outputs
            self.weights += learning_rate * np.dot(error, inputs)
            self.bias += learning_rate * np.sum(error)

    def forward_pass(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

    def backward_pass(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        error = targets - outputs
        return error * self.sigmoid_derivative(outputs)