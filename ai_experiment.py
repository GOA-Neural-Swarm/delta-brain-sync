import numpy as np
from sklearn.preprocessing import normalize

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.weights = np.random.rand(neurons, synapses)

    def simulate(self, inputs):
        activations = np.zeros((self.neurons,))
        for i in range(self.neurons):
            for j in range(self.synapses):
                activations[i] += inputs[j] * self.weights[i, j]
        activations = normalize(activations)
        return activations