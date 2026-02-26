import time
import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.neurons = np.zeros((num_neurons, 1), dtype=float)
        self.synapses = np.zeros((num_synapses, num_neurons), dtype=float)

    def process(self, input_data):
        self.neurons = np.dot(self.synapses.T, input_data)
        return self.neurons