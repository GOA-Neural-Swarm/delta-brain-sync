import numpy as np
import numba

@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights_ih = np.random.rand(num_inputs, num_hidden)
        self.weights_ho = np.random.rand(num_hidden, num_outputs)
        self.biases_ih = np.zeros((num_hidden,))
        self.biases_ho = np.zeros((num_outputs,))

    def feedforward(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_ih) + self.biases_ih
        hidden_layer = sigmoid(hidden_layer)
        outputs = np.dot(hidden_layer, self.weights_ho) + self.biases_ho
        outputs = sigmoid(outputs)
        return outputs

    def train(self, inputs, targets, learning_rate, num_iterations):
        for _ in range(num_iterations):
            hidden_layer = np.dot(inputs, self.weights_ih) + self.biases_ih
            hidden_layer = sigmoid(hidden_layer)
            outputs = np.dot(hidden_layer, self.weights_ho) + self.biases_ho
            outputs = sigmoid(outputs)
            error = targets - outputs
            deltas = error * outputs * (1 - outputs)
            self.weights_ho += np.dot(hidden_layer.T, deltas) * learning_rate
            self.biases_ho += np.sum(deltas, axis=0, keepdims=True) * learning_rate
            error = np.dot(deltas, self.weights_ho.T) * hidden_layer * (1 - hidden_layer)
            self.weights_ih += np.dot(inputs.T, error) * learning_rate
            self.biases_ih += np.sum(error, axis=0, keepdims=True) * learning_rate