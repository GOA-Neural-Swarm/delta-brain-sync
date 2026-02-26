import numpy as np

class Brain:
    def __init__(self):
        self.neural_network = {}
        self.synapses = {}

    def connect(self, neuron1, neuron2, strength):
        if neuron1 not in self.neural_network:
            self.neural_network[neuron1] = {}
        if neuron2 not in self.neural_network:
            self.neural_network[neuron2] = {}
        self.neural_network[neuron1][neuron2] = strength
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        if neuron2 not in self.synapses:
            self.synapses[neuron2] = {}
        self.synapses[neuron1][neuron2] = strength

    def process(self, inputs):
        for neuron, value in inputs.items():
            if neuron not in self.neural_network:
                self.neural_network[neuron] = {}
            for connected_neuron, strength in self.neural_network[neuron].items():
                self.neural_network[connected_neuron].setdefault(neuron, 0)
                self.neural_network[connected_neuron][neuron] += value * strength
        for neuron, values in self.neural_network.items():
            if neuron not in self.synapses:
                self.synapses[neuron] = {}
            for connected_neuron, strength in self.synapses[neuron].items():
                self.neural_network[connected_neuron].setdefault(neuron, 0)
                self.neural_network[connected_neuron][neuron] += self.neural_network[neuron][connected_neuron] * strength

    def get_output(self):
        output = {}
        for neuron, values in self.neural_network.items():
            output[neuron] = np.sum([value * strength for value, strength in values.items()])
        return output