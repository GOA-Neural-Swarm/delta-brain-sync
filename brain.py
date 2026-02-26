import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1)
        self.bias = np.random.rand(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        for _ in range(1000):
            z = np.dot(X, self.weights) + self.bias
            layer_outputs = self.sigmoid(z)
            predictions = layer_outputs
            loss = np.mean((predictions - y) ** 2)
            dloss = 2 * (predictions - y)
            dlayer_outputs = dloss
            delta = dlayer_outputs * self.sigmoid_derivative(layer_outputs)
            self.weights += 0.1 * np.dot(X.T, delta)
            self.bias += 0.1 * delta
        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)