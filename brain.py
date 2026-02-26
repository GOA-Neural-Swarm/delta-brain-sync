import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)  # optimized for high-speed processing
        self.biases = np.zeros((100, 1))
        self.input_size = 100
        self.hidden_size = 100
        self.output_size = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # optimized for high-speed processing

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # optimized for high-speed processing

    def forward_propagation(self, inputs):
        self.input_layer = inputs
        self.hidden_layer = np.dot(self.weights, self.input_layer) + self.biases
        self.output_layer = self.sigmoid(self.hidden_layer)
        return self.output_layer

    def backward_propagation(self, inputs, targets):
        self.input_layer = inputs
        self.hidden_layer = np.dot(self.weights, self.input_layer) + self.biases
        self.output_layer = self.sigmoid(self.hidden_layer)
        self.error = targets - self.output_layer
        self.weights += np.dot(self.error, self.hidden_layer.T) * 0.1
        self.biases += np.sum(self.error, axis=0, keepdims=True) * 0.1
        return self.error

    def train(self, inputs, targets):
        self.error = np.zeros((self.output_size, 1))
        for _ in range(1000):  # optimized for high-speed processing
            self.error = self.backward_propagation(inputs, targets)
            if np.mean(self.error) < 0.01:  # optimized for high-speed processing
                break
        return self.error