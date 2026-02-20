import numpy as np
import matplotlib.pyplot as plt

# Define neural network architecture
class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)
        self.biases = np.zeros((outputs,))

    def forward_pass(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def backward_pass(self, inputs, targets):
        errors = targets - self.forward_pass(inputs)
        self.weights += np.dot(inputs.T, errors)
        self.biases += np.sum(errors, axis=0, keepdims=True)

    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            for i in range(len(inputs)):
                self.backward_pass(inputs[i], targets[i])

    def predict(self, inputs):
        return self.forward_pass(inputs)

# Load sequence data
sequence_data = np.array([int(i) for i in Source])

# Initialize neural network
nn = NeuralNetwork(20, 1)

# Train neural network
nn.train(sequence_data, np.array([1] * len(sequence_data)), 1000)

# Predict output
output = nn.predict(sequence_data)

print(output)