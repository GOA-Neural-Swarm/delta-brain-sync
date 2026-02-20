import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.bias = np.zeros((10,))

    def think(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def learn(self, inputs, outputs):
        error = outputs - self.think(inputs)
        self.weights += error * np.dot(inputs.T, error)
        self.bias += error

brain = Brain()
brain.learn(np.array([[1, 2, 3, 4, 5]]), np.array([1, 2, 3, 4, 5]))