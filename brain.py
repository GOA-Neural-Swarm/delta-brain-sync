import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.synaptic_weights = np.random.rand(num_synapses)

    def compute(self, inputs):
        outputs = np.zeros(self.num_neurons)
        for i in range(self.num_synapses):
            outputs += self.synaptic_weights[i] * inputs[i]
        return outputs

    def mutate(self, mutation_rate):
        for i in range(self.num_synapses):
            if np.random.rand() < mutation_rate:
                self.synaptic_weights[i] = np.random.rand()