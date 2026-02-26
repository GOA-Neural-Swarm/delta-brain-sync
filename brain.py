import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)
        self.biases = np.random.rand(100, 1)
        self.output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(self.weights, inputs) + self.biases)

    def backward_pass(self, targets):
        self.targets = targets
        error = np.mean((self.output - self.targets) ** 2)
        d_output = 2 * (self.output - self.targets)
        d_weights = np.dot(d_output, self.inputs.T)
        d_biases = d_output
        return error, d_weights, d_biases

    def update_weights(self, error, d_weights, d_biases, learning_rate):
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases