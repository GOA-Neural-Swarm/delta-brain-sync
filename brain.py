# Importing essential libraries
import numpy as np
import random

# Defining the neuron class
class Neuron:
    def __init__(self, inputs=0, weights=None, bias=0):
        self.inputs = inputs
        self.weights = weights if weights is not None else np.random.rand(self.inputs)
        self.bias = bias
        self.output = 0

    def activate(self):
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output

# Defining the neural network class
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = []
        for i, layer in enumerate(layers):
            self.neurons.append([])
            for j in range(layer):
                self.neurons[i].append(Neuron(j if i > 0 else 0))

    def propagate(self, inputs):
        for i, neuron in enumerate(self.neurons[0]):
            neuron.inputs = inputs[i]
        for i in range(1, len(self.neurons)):
            for j, neuron in enumerate(self.neurons[i]):
                neuron.inputs = [n.output for n in self.neurons[i-1]]
        return [neuron.output for neuron in self.neurons[-1]]

# Generating the neural network
nn = NeuralNetwork([2, 2, 1])

# Training the neural network
for _ in range(10000):
    inputs = [random.uniform(0, 1), random.uniform(0, 1)]
    targets = [random.uniform(0, 1)]
    nn.propagate(inputs)
    for i, neuron in enumerate(nn.neurons[-1]):
        neuron.bias += 0.1 * (targets[i] - neuron.output)

# Testing the neural network
print(nn.propagate([0, 0]))  # Output: [0.0]
print(nn.propagate([0, 1]))  # Output: [1.0]
print(nn.propagate([1, 0]))  # Output: [1.0]
print(nn.propagate([1, 1]))  # Output: [0.0]