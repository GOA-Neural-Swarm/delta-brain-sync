import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1000, 1000)
        self.biases = np.random.rand(1000)

    def process(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return np.where(self.outputs > 0.5, 1, 0)

    def mutate(self):
        self.weights += np.random.normal(0, 0.1, size=self.weights.shape)
        self.biases += np.random.normal(0, 0.1, size=self.biases.shape)