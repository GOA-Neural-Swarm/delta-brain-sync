import numpy as np
from numba import jit

class Brain:
    def __init__(self):
        self.synapses = {}

    def connect(self, neuron1, neuron2):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        self.synapses[neuron1][neuron2] = np.random.rand()

    @jit(nopython=True)
    def propagate(self, inputs):
        outputs = {}
        for neuron, value in inputs.items():
            outputs[neuron] = 0
            for synapse, strength in self.synapses.get(neuron, {}).items():
                outputs[neuron] += strength * value
        return outputs