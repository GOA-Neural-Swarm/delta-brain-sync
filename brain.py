import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}

    def connect(self, neuron1, neuron2):
        if (neuron1, neuron2) in self.synapses:
            self.synapses[(neuron1, neuron2)]['weight'] += 1
        else:
            self.synapses[(neuron1, neuron2)] = {'weight': 1}

    def fire(self, neuron):
        fired_synapses = []
        for synapse in self.synapses:
            if synapse[0] == neuron:
                fired_synapses.append(synapse)
        for synapse in fired_synapses:
            self.connect(synapse[0], synapse[1])
        return np.sum([synapse['weight'] for synapse in fired_synapses])

    def run(self, neurons):
        results = []
        for neuron in neurons:
            results.append(self.fire(neuron))
        return results