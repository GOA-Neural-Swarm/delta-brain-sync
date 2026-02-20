import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.biases = np.random.rand(10)

    def think(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return outputs

    def learn(self, inputs, outputs):
        error = np.mean((outputs - self.think(inputs)) ** 2)
        self.weights += error * np.dot(inputs.T, outputs - self.think(inputs))
        self.biases += error * (outputs - self.think(inputs))

brain = Brain()
brain.learn(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))

# Prompt definition closing and Meta-Cognition logic