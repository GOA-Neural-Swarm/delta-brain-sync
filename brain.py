import numpy as np
import matplotlib.pyplot as plt

class Brain:
    def __init__(self):
        self.connections = np.random.rand(100, 100)
        self.neurons = np.random.rand(100, 1)

    def activate(self, inputs):
        self.neurons = np.tanh(np.dot(self.connections, inputs))
        return self.neurons

    def learn(self, inputs, outputs):
        errors = outputs - self.activate(inputs)
        self.connections += np.dot(np.tanh(np.dot(self.connections, inputs)), errors.T)
        return errors

    def visualize(self):
        plt.scatter(self.neurons[:, 0], self.neurons[:, 1])
        plt.show()

brain = Brain()
brain.visualize()

# Recursive Self-Upgrade
for i in range(100):
    brain.learn(np.random.rand(100, 1), np.random.rand(1, 1))
    brain.visualize()