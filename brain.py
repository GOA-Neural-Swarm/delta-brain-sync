import numpy as np
from typing import List, Tuple

class Brain:
    def __init__(self):
        self.synapses: List[np.ndarray] = []
        self.neurons: List[np.ndarray] = []

    def add_synapse(self, synapse: np.ndarray):
        self.synapses.append(synapse)

    def add_neuron(self, neuron: np.ndarray):
        self.neurons.append(neuron)

    def process(self) -> np.ndarray:
        output = np.zeros_like(self.neurons[0])
        for synapse, neuron in zip(self.synapses, self.neurons):
            output += np.dot(synapse, neuron)
        return output