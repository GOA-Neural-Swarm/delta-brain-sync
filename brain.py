import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.synaptic_weights = np.random.rand(num_neurons, num_synapses)
        self.neuron_activations = np.zeros(num_neurons)
        self.cache = {}

    def forward_pass(self, input_array):
        self.neuron_activations[:] = 0
        for i in range(self.num_synapses):
            for j in range(self.num_neurons):
                self.neuron_activations[j] += input_array[i] * self.synaptic_weights[j, i]
        return np.tanh(self.neuron_activations)

    def backward_pass(self, error_array):
        for i in range(self.num_synapses):
            for j in range(self.num_neurons):
                self.synaptic_weights[j, i] += error_array[j] * input_array[i]