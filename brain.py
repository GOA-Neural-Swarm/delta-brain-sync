import numpy as np
from collections import defaultdict
class Brain:
    def __init__(self):
        self.neurons = defaultdict(dict)
        self.synapses = defaultdict(dict)
    
    def add_neuron(self, neuron_id):
        self.neurons[neuron_id]['inputs'] = []
        self.neurons[neuron_id]['output'] = 0
    
    def add_synapse(self, neuron_id, synapse_id):
        self.synapses[(neuron_id, synapse_id)] = {'weight': 0, 'delay': 0}
    
    def fire_neuron(self, neuron_id):
        neuron = self.neurons[neuron_id]
        if neuron['inputs']:
            neuron['output'] = np.sum([synapse['weight'] * self.get_synapse_output(synapse['delay']) for synapse in neuron['inputs']])
        else:
            neuron['output'] = 0
    
    def get_synapse_output(self, delay):
        return self.synapses.get((delay, delay), {'weight': 0, 'delay': 0})['output']