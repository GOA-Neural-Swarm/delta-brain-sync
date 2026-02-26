from collections import defaultdict
import math

class Brain:
    def __init__(self):
        self.memory = defaultdict(dict)
        self.synapses = defaultdict(dict)
        self.weights = defaultdict(dict)

    def process(self, input_data):
        for neuron, value in input_data.items():
            for connected_neuron, weight in self.weights[neuron].items():
                self.synapses[neuron][connected_neuron] = value * weight
            self.memory[neuron] = self.synapses[neuron]
            self.synapses[neuron] = defaultdict(dict)

    def retrieve(self, neuron):
        return self.memory[neuron].copy()

    def update_weights(self, neuron, connected_neuron, weight):
        self.weights[neuron][connected_neuron] = weight
        self.synapses[neuron][connected_neuron] = self.memory[neuron][connected_neuron] * weight