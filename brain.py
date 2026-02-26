import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.neurons = np.random.rand(num_neurons)
        self.synapses = np.random.rand(num_synapses)

    def process(self, input_data):
        for i in range(self.num_neurons):
            self.neurons[i] += np.sum(self.synapses[i] * input_data)
        return self.neurons