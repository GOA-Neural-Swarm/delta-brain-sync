import numpy as np

class Brain:
    def __init__(self):
        self.synaptic_weights = np.random.rand(3, 3)
        self.threshold = 0.5

    def think(self, inputs):
        output = np.dot(inputs, self.synaptic_weights)
        return np.where(output > self.threshold, 1, 0)

    def learn(self, inputs, output):
        adjustments = np.dot(inputs.T, output - self.think(inputs))
        self.synaptic_weights += adjustments