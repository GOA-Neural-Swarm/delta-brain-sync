import numpy as np
import numba

@numba.jit(nopython=True)
class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)

    @numba.jit(nopython=True)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @numba.jit(nopython=True)
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    @numba.jit(nopython=True)
    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)
        return output_layer

    @numba.jit(nopython=True)
    def train(self, inputs, targets, learning_rate):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)

        # Calculate the error
        errors = targets - output_layer

        # Calculate the hidden layer errors
        hidden_layer_errors = np.dot(errors, self.weights2.T)
        hidden_layer_errors *= self.sigmoid_derivative(hidden_layer)

        # Calculate the weights update
        self.weights1 += learning_rate * np.dot(inputs.T, hidden_layer_errors)
        self.weights2 += learning_rate * np.dot(hidden_layer.T, errors)