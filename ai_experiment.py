import numpy as np
from numba import njit

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons
        self.synapses = np.random.rand(neurons, neurons)

    @njit
    def process(self, inputs):
        outputs = np.zeros(self.neurons)
        for i in range(self.neurons):
            for j in range(self.neurons):
                outputs[i] += self.synapses[i, j] * inputs[j]
        return outputs