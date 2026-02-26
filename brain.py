import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1, 1)  # weights for synaptic connection
        self.bias = np.random.rand(1)  # bias for neuron activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # sigmoid activation function

    def forward_pass(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.sigmoid(weighted_sum)
        return output

    def train(self, inputs, targets):
        predicted_outputs = self.forward_pass(inputs)
        error = targets - predicted_outputs
        self.weights += 0.1 * np.dot(inputs.T, error)
        self.bias += 0.1 * error.mean()