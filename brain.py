import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10)
        self.biases = np.random.rand(10)

    def process(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return np.where(outputs > 0.5, 1, 0)