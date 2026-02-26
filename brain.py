import numpy as np

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.memory = np.zeros((neurons, neurons))

    def fire(self, neuron_id):
        neuron = self.neurons[neuron_id]
        if neuron['active']:
            for synapse_id, weight in self.synapses[neuron_id].items():
                self.memory[neuron_id][synapse_id] += weight * neuron['output']
            neuron['active'] = False

    def learn(self):
        for i, _ in enumerate(self.memory):
            for j, _ in enumerate(self.memory[i]):
                self.memory[i][j] /= self.memory[i].sum()