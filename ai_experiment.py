import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1, 1)
        self.biases = np.random.rand(1)
        self.activity_threshold = 0.5

    def activate(self, input_array):
        return np.where(input_array > self.activity_threshold, 1, 0)

    def forward_pass(self, inputs):
        return self.activate(np.dot(inputs, self.weights) + self.biases)

    def backward_pass(self, outputs, expected_outputs):
        error = np.sum((outputs - expected_outputs) ** 2) / 2
        self.weights -= 0.1 * np.dot(np.transpose(outputs), outputs - expected_outputs)
        self.biases -= 0.1 * np.sum(outputs - expected_outputs)
        return error