import numpy as np
import random

class Neuron:
    def __init__(self, weights=None, bias=0):
        self.weights = weights if weights is not None else np.random.rand(1)
        self.bias = bias

    def activate(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = [Neuron() for _ in range(layers)]

    def forward_pass(self, inputs):
        for i in range(len(self.layers) - 1):
            inputs = [self.layers[i].activate([inputs[j]]) for j in range(len(inputs))]
        return [self.layers[-1].activate([inputs[j]]) for j in range(len(inputs))]

# Prompt definition closing and Meta-Cognition logic
if __name__ == "__main__":
    # Initialize neural network with 2 inputs, 1 hidden layer with 3 neurons, and 1 output
    network = NeuralNetwork([2, 3, 1])

    # Define input sequence
    inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    # Forward pass
    outputs = network.forward_pass(inputs)

    print("Predicted outputs:", outputs)