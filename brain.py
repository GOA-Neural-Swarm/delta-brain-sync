import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.synaptic_weights = np.random.rand(num_synapses).flatten()

    def process(self, input_values):
        return np.dot(input_values, self.synaptic_weights)