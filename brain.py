import numpy as np
from numba import jit

class Brain:
    def __init__(self):
        self.synapses = {}

    @jit(nopython=True)
    def process(self, inputs):
        outputs = np.zeros((len(inputs),))
        for i, input in enumerate(inputs):
            if i in self.synapses:
                outputs[i] = self.synapses[i] * input
            else:
                outputs[i] = input
        return outputs

    def train(self, inputs, outputs):
        for i, input in enumerate(inputs):
            if i not in self.synapses:
                self.synapses[i] = np.mean((outputs[i] - input) / np.std((outputs[i] - input)))