import numpy as np

class Brain:
    def __init__(self):
        self.synaptic_weights = np.random.rand(4, 4)

    def think(self, inputs):
        return np.dot(inputs, self.synaptic_weights)

    def train(self, inputs, outputs):
        self.synaptic_weights += np.dot(inputs.T, outputs - self.think(inputs)) / inputs.shape[0]

    def mutate(self):
        self.synaptic_weights += np.random.normal(0, 0.1, size=self.synaptic_weights.shape)

    def crossover(self, other):
        mask = np.random.rand(*self.synaptic_weights.shape) < 0.5
        self.synaptic_weights[mask] = other.synaptic_weights[mask]

    def selection(self, brains):
        scores = [brain.think(inputs) for brain in brains]
        return brains[np.argmax(scores)]