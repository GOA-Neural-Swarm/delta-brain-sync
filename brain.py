import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(*layer) for layer in layers[1:]]

    def forward_pass(self, inputs):
        activations = [inputs]
        for i in range(len(self.layers[1:]) - 1):
            z = np.dot(activations[-1], self.weights[i])
            activations.append(self._sigmoid(z))
        return activations[-1]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            inputs = np.array(inputs)
            targets = np.array(targets)
            outputs = self.forward_pass(inputs)
            errors = targets - outputs
            self.weights = [self.weights[i] + np.dot(activations[-1], errors) for i, activations in enumerate(zip(*[self.layers[1:]]))]
        return self

# Prompt definition and Meta-Cognition logic