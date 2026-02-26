class Brain:
    def __init__(self):
        self.neurons = {}
        self.synapses = {}

    def add_neuron(self, neuron_id):
        self.neurons[neuron_id] = {'activation': 0, 'weight': 0}

    def add_synapse(self, neuron_id1, neuron_id2, weight):
        if neuron_id1 not in self.synapses:
            self.synapses[neuron_id1] = {}
        if neuron_id2 not in self.synapses:
            self.synapses[neuron_id2] = {}
        self.synapses[neuron_id1][neuron_id2] = weight

    def process(self):
        for neuron_id, neuron_data in self.neurons.items():
            neuron_data['activation'] = sum(self.synapses.get(neuron_id, {}).get(neuron_id2, 0) * neuron_data['activation'] for neuron_id2 in self.synapses.get(neuron_id, {}))
            self.neurons[neuron_id]['activation'] %= 1