import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1000, 1000)
        self.biases = np.random.rand(1000)
        self.input_neurons = np.zeros(1000)
        self.output_neurons = np.zeros(1000)

    def forward_propagate(self, inputs):
        self.input_neurons = inputs
        self.output_neurons = np.dot(self.input_neurons, self.weights.T) + self.biases
        return self.output_neurons

    def backward_propagate(self, targets, learning_rate=0.01):
        output_errors = targets - self.output_neurons
        weight_deltas = np.dot(output_errors.T, self.input_neurons)
        bias_deltas = output_errors
        self.weights -= learning_rate * weight_deltas
        self.biases -= learning_rate * bias_deltas
        return weight_deltas, bias_deltas

    def train(self, inputs, targets, learning_rate=0.01, iterations=1000):
        for _ in range(iterations):
            self.forward_propagate(inputs)
            weight_deltas, bias_deltas = self.backward_propagate(targets, learning_rate)
        return self.output_neurons