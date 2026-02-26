import numpy as np

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, neurons)
        self.biases = np.random.rand(neurons)

    def forward_pass(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return outputs

    def backward_pass(self, errors):
        weights_update = np.dot(errors, self.neurons)
        biases_update = errors
        return weights_update, biases_update

    def optimize(self, inputs, outputs):
        errors = np.abs(outputs - self.forward_pass(inputs))
        weights_update, biases_update = self.backward_pass(errors)
        self.weights -= 0.01 * weights_update
        self.biases -= 0.01 * biases_update