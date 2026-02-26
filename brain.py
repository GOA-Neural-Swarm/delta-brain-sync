import numpy as np
import numba

@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@numba.jit(nopython=True)
def sigmoid_derivative(x):
    return x * (1 - x)

class Brain:
    def __init__(self, inputs, outputs):
        self.input_shape = inputs
        self.output_shape = outputs
        self.weights = np.random.rand(self.output_shape[0], self.input_shape[0])
        self.biases = np.zeros((self.output_shape[0],))

    def forward_pass(self, inputs):
        inputs = np.array(inputs)
        hidden_layer = sigmoid(np.dot(inputs, self.weights) + self.biases)
        output = sigmoid(np.dot(hidden_layer, self.weights) + self.biases)
        return output

    def train(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        hidden_layer = sigmoid(np.dot(inputs, self.weights) + self.biases)
        output = sigmoid(np.dot(hidden_layer, self.weights) + self.biases)
        error = targets - output
        delta = error * sigmoid_derivative(output)
        self.weights += np.dot(delta, hidden_layer.T)
        self.biases += np.sum(delta, axis=0, keepdims=True)