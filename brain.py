class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.deltas = [0] * len(neurons)

    def propagate(self, inputs):
        for i, neuron in enumerate(self.neurons):
            neuron['output'] = neuron.get('threshold', 0)  # cache threshold
            for j, synapse in enumerate(self.synapses[i]):
                neuron['output'] += synapse['weight'] * inputs[j]
            if neuron['output'] > neuron.get('threshold', 0):
                neuron['output'] = 1
            else:
                neuron['output'] = 0

    def update(self, targets):
        for i, neuron in enumerate(self.neurons):
            error = targets[i] - neuron['output']
            self.deltas[i] = error
            for j, synapse in enumerate(self.synapses[i]):
                synapse['weight'] += error * synapse['learning_rate'] * neuron['output']

    def __repr__(self):
        return f'Brain(neurons={len(self.neurons)}, synapses={len(self.synapses)})'