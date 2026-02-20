import numpy as np

class OptimizedBrain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = np.random.rand(100, 100)
        self.biases = np.random.rand(100)
        self.learning_rate = 0.1

    def calculate_activation(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases

    def propagate_error(self, expected_output):
        error = expected_output - self.calculate_activation(self.sequence)
        return error

    def update_weights_biases(self, error):
        self.weights += self.learning_rate * np.dot(error[:, np.newaxis], self.sequence)[np.newaxis, :]
        self.biases += self.learning_rate * error

    def train(self, sequence, expected_output):
        activation = self.calculate_activation(sequence)
        error = self.propagate_error(expected_output)
        self.update_weights_biases(error)

    def run(self):
        self.train(self.sequence, self.sequence)

brain = OptimizedBrain(Source)
brain.run()