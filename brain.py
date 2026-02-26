import numpy as np
from scipy import ndimage

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights_ih = np.random.rand(num_inputs, num_hidden)
        self.weights_ho = np.random.rand(num_hidden, num_outputs)
        self.biases_ih = np.zeros((1, num_hidden))
        self.biases_ho = np.zeros((1, num_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_ih) + self.biases_ih
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights_ho) + self.biases_ho
        output_layer = self.sigmoid(output_layer)
        return output_layer

    def backward_pass(self, inputs, targets):
        hidden_layer = self.forward_pass(inputs)
        output_layer = self.sigmoid_derivative(hidden_layer)
        output_error = targets - hidden_layer
        hidden_error = np.dot(output_error, self.weights_ho.T) * self.sigmoid_derivative(hidden_layer)
        self.weights_ho += np.dot(hidden_layer.T, output_error)
        self.weights_ih += np.dot(inputs.T, hidden_error)
        self.biases_ho += output_error
        self.biases_ih += hidden_error

    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            self.backward_pass(inputs, targets)