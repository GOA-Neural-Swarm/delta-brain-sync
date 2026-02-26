import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.biases = np.random.rand(10)
        self.cache = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        self.cache.append(inputs)
        return np.dot(inputs, self.weights) + self.biases

    def backward_pass(self):
        gradients = []
        for i in range(len(self.cache) - 1):
            x = self.cache[-1 - i - 1]
            y = self.cache[-1 - i]
            gradients.append(np.dot(y.T, x))
        return gradients[::-1]

    def optimize(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        errors = targets - outputs
        gradients = self.backward_pass()
        self.weights -= 0.1 * np.dot(self.cache[0].T, errors)
        self.biases -= 0.1 * np.sum(errors, axis=0)
        return outputs