import random
import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.biases = np.zeros((10,))
        self.neurons = np.zeros((10,))

    def propagate(self, inputs):
        self.neurons[0] = inputs[0]
        for i in range(1, 10):
            self.neurons[i] = np.tanh(np.dot(self.weights[i-1], inputs) + self.biases[i-1])
        return self.neurons

    def think(self, inputs):
        return np.argmax(self.propagate(inputs))

brain = Brain()
print(brain.think([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))