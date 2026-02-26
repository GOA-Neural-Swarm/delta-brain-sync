import numpy as np

class Brain:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.random.rand(layers[i+1], 1) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        outputs = inputs
        for i in range(len(self.layers)-1):
            outputs = self.sigmoid(np.dot(self.weights[i], outputs) + self.biases[i])
        return outputs

    def backpropagate(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        errors = targets - outputs
        for i in range(len(self.layers)-1, 0, -1):
            errors = errors * self.sigmoid_derivative(outputs)
            outputs = np.dot(self.weights[i-1].T, errors)
        return errors