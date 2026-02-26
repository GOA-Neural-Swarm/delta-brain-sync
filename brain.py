import numpy as np
import numba

@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@numba.jit(nopython=True)
def sigmoid_derivative(x):
    return x * (1 - x)

class Brain:
    def __init__(self, neurons, layers):
        self.layers = layers
        self.neurons = neurons
        self.synaptic_weights = np.random.uniform(size=(neurons, neurons))

    @numba.jit(nopython=True)
    def think(self, inputs):
        for layer in range(self.layers - 1):
            inputs = sigmoid(np.dot(inputs, self.synaptic_weights[:, layer]))
        return sigmoid(np.dot(inputs, self.synaptic_weights[:, self.layers - 1]))

    @numba.jit(nopython=True)
    def train(self, inputs, targets, iterations):
        for _ in range(iterations):
            output = self.think(inputs)
            error = targets - output
            delta = error * sigmoid_derivative(output)
            for layer in range(self.layers - 1, 0, -1):
                delta = delta * sigmoid_derivative(output)
                output = np.dot(delta, self.synaptic_weights[:, layer])
                self.synaptic_weights[:, layer] += 0.1 * np.dot(inputs.T, output)