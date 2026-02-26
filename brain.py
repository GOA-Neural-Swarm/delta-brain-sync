class Brain:
    def __init__(self):
        self.neurons = {}
        self.synapses = {}

    def add_neuron(self, name):
        self.neurons[name] = {'inputs': [], 'outputs': []}

    def add_synapse(self, from_neuron, to_neuron):
        self.synapses.setdefault(from_neuron, []).append(to_neuron)
        self.synapses.setdefault(to_neuron, []).append(from_neuron)

    def fire(self, neuron):
        if neuron in self.neurons:
            outputs = self.neurons[neuron]['outputs']
            for synapse in self.synapses.get(neuron, []):
                if synapse in self.neurons:
                    outputs.extend(self.neurons[synapse]['outputs'])
            return outputs
        return []