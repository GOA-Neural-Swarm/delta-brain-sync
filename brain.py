import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)  # Initialize weights with random values
        self.biases = np.random.rand(100)  # Initialize biases with random values

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        # Forward pass
        outputs_pred = self.forward_pass(inputs)
        error = np.mean((outputs_pred - outputs) ** 2)

        # Backward pass
        deltas = 2 * (outputs_pred - outputs)
        hidden_layer_error = np.dot(deltas, self.weights.T)
        self.weights += np.dot(inputs.T, hidden_layer_error)
        self.biases += np.sum(hidden_layer_error, axis=0)

    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.weights) + self.biases
        return self.sigmoid(hidden_layer)