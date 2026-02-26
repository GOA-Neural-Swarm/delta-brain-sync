from collections import defaultdict
from numba import jit, types
import numpy as np

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.cache = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0

    @jit
    def process(self, inputs):
        if inputs in self.cache:
            self.cache_hits += 1
            return self.cache[inputs]
        else:
            self.cache_misses += 1
            output = self._calculate_output(inputs)
            self.cache[inputs] = output
            return output

    @jit
    def _calculate_output(self, inputs):
        output = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            for synapse in self.synapses:
                if synapse['input'] == i:
                    output[i] += neuron * synapse['weight']
        return output