import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1000, 1000)
        self.biases = np.random.rand(1000, 1)
        self.output = np.zeros((1000, 1))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backpropagation(self, inputs, targets):
        output = self.forward_pass(inputs)
        error = np.sum((output - targets) ** 2) / 2
        delta_output = (output - targets)
        delta_weights = np.dot(delta_output.T, inputs)
        delta_biases = np.sum(delta_output, axis=0, keepdims=True)
        return error, delta_weights, delta_biases

    def train(self, inputs, targets, learning_rate=0.01):
        error, delta_weights, delta_biases = self.backpropagation(inputs, targets)
        self.weights -= learning_rate * delta_weights
        self.biases -= learning_rate * delta_biases
        return error