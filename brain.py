import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.neurons = np.random.rand(num_neurons)
        self.synapses = np.random.rand(num_synapses)
        self.weights = np.random.rand(num_neurons, num_synapses)

    def fire(self, input_signals):
        outputs = np.dot(input_signals, self.weights)
        self.neurons += outputs
        self.synapses *= self.synapses
        return self.neurons