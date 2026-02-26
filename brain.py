import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1, 10)
        self.biases = np.random.rand(10)
        self.cache = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        if tuple(inputs) in self.cache:
            return self.cache[tuple(inputs)]
        outputs = np.dot(inputs, self.weights) + self.biases
        self.cache[tuple(inputs)] = outputs
        return self.sigmoid(outputs)

    def backward_pass(self, inputs, expected_output):
        output = self.forward_pass(inputs)
        output_derivative = self.sigmoid_derivative(output)
        weight_update = inputs * output_derivative
        bias_update = output_derivative
        self.weights += 0.1 * weight_update
        self.biases += 0.1 * bias_update
        return output_derivative

    def train(self, inputs, expected_outputs):
        for inputs, expected_output in zip(inputs, expected_outputs):
            output_derivative = self.backward_pass(inputs, expected_output)
            return output_derivative

    def predict(self, inputs):
        return self.forward_pass(inputs)