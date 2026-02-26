import numba
import numpy as np

@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Brain:
    def __init__(self):
        self.synapses = {}

    def process(self, inputs):
        outputs = {}
        for input_key, input_value in inputs.items():
            if input_key in self.synapses:
                output_value = self.synapses[input_key](input_value)
                outputs[input_key] = output_value
            else:
                outputs[input_key] = input_value
        return outputs

    def add_synapse(self, input_key, input_function):
        self.synapses[input_key] = input_function

    def optimize_synapses(self):
        for key, value in self.synapses.items():
            self.synapses[key] = numba.jit(value)