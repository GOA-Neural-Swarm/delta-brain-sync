import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.neurons = np.zeros((num_neurons, num_synapses))
        self.synapses = np.random.rand(num_synapses)

    def fire(self, inputs):
        outputs = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            outputs[i] = np.sum(self.synapses * self.neurons[i])
        return outputs

    def learn(self, inputs, outputs):
        for i in range(self.num_neurons):
            self.synapses[i] += np.dot(self.neurons[i], outputs[i]) * 0.1