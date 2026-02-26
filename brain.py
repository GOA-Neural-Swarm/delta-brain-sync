import numpy as np

class Brain:
    def __init__(self):
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        self.learning_rate = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        output = np.dot(inputs, self.synaptic_weights)
        return self.sigmoid(output)

    def train(self, inputs, output):
        output = self.think(inputs)
        error = output - output
        adjustment = error * self.derivative(output) * self.learning_rate
        self.synaptic_weights += adjustment