import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}

    def connect(self, neuron1, neuron2):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        self.synapses[neuron1][neuron2] = np.random.rand()

    def fire(self, neuron):
        if neuron in self.synapses:
            total_weight = sum(self.synapses[neuron].values())
            return total_weight > np.random.rand()
        return False

    def stimulate(self, neuron):
        if neuron in self.synapses:
            for other_neuron, weight in self.synapses[neuron].items():
                if self.fire(other_neuron):
                    self.stimulate(other_neuron)
        return