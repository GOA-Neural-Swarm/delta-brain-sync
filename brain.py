import random
import math
import numpy as np

class Brain:
    def __init__(self):
        self.synaptic_weights = np.random.rand(10, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def propagate(self, inputs, weights):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(inputs, weights))

    def backpropagate(self, inputs, weights, targets):
        self.propagate(inputs, weights)
        error = targets - self.output
        self.error = error
        delta = error * self.sigmoid_derivative(self.output)
        weights += np.dot(np.transpose(self.inputs), delta)

    def train(self, inputs, targets):
        for _ in range(1000):
            self.backpropagate(inputs, self.synaptic_weights, targets)

    def think(self, inputs):
        self.propagate(inputs, self.synaptic_weights)
        return self.output

brain = Brain()
brain.train(np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1]]), np.array([[0], [1], [1]]))

# Prompt definition closing and Meta-Cognition logic
print(brain.think(np.array([1, 0, 1])))