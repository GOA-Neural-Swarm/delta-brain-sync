import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.biases = np.random.rand(10)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)

        for i in range(1000):
            hidden_layer = np.dot(inputs, self.weights) + self.biases
            outputs = self.sigmoid(hidden_layer)

            error = targets - outputs
            self.weights += np.dot(inputs.T, error * self.sigmoid_derivative(outputs))
            self.biases += error.mean()

    def think(self, inputs):
        inputs = np.array(inputs)
        hidden_layer = np.dot(inputs, self.weights) + self.biases
        return self.sigmoid(hidden_layer)