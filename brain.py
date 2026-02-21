import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, neurons, layers):
        self.neurons = neurons
        self.layers = layers
        self.weights = [np.random.rand(neurons[i], neurons[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((neurons[i+1],)) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.weights[0]) + self.biases[0]
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights[1]) + self.biases[1]
        output_layer = self.sigmoid(output_layer)
        return output_layer

    def backpropagation(self, inputs, targets):
        hidden_layer = self.forward_pass(inputs)
        output_layer = self.forward_pass(hidden_layer)
        error = np.sum((output_layer - targets) ** 2)
        delta_output = 2 * (output_layer - targets) * self.sigmoid_derivative(output_layer)
        delta_hidden = np.dot(delta_output, self.weights[1].T) * self.sigmoid_derivative(hidden_layer)
        self.weights[1] -= np.dot(hidden_layer.T, delta_output)
        self.biases[1] -= np.sum(delta_output, axis=0, keepdims=True)
        self.weights[0] -= np.dot(inputs.T, delta_hidden)
        self.biases[0] -= np.sum(delta_hidden, axis=0, keepdims=True)

# Testing the neural network
nn = NeuralNetwork(neurons=[2, 4, 1], layers=3)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])
nn.backpropagation(inputs, targets)
print(nn.forward_pass(inputs))