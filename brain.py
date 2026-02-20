import random
import math

class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def add_synapse(self, neuron1, neuron2, weight):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        self.synapses[neuron1][neuron2] = weight

    def add_neuron(self, neuron, inputs):
        self.neurons[neuron] = inputs

    def process_neuron(self, neuron):
        if neuron not in self.synapses:
            return 0
        total_weight = 0
        for other_neuron, weight in self.synapses[neuron].items():
            total_weight += weight * self.process_neuron(other_neuron)
        return total_weight

    def evolve(self):
        new_synapses = {}
        new_neurons = {}
        for neuron, inputs in self.neurons.items():
            new_inputs = []
            for input_neuron, weight in inputs.items():
                new_inputs.append((input_neuron, weight * math.exp(-random.uniform(0, 1))))
            new_synapses[neuron] = new_inputs
        self.synapses = new_synapses
        self.neurons = new_neurons

brain = Brain()
brain.add_synapse('A', 'B', 0.5)
brain.add_synapse('B', 'C', 0.7)
brain.add_synapse('C', 'A', 0.3)
brain.add_neuron('A', [('B', 0.5)])
brain.add_neuron('B', [('A', 0.5), ('C', 0.7)])
brain.add_neuron('C', [('B', 0.7)])
print(brain.process_neuron('A'))
brain.evolve()
print(brain.process_neuron('A'))