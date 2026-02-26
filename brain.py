import numba
import numpy as np

@numba.jit(nopython=True)
def process(input_data):
    return np.dot(input_data, np.random.rand(input_data.shape[1], 1))

class Brain:
    def __init__(self):
        self.layers = []
        self.optimizers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_optimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def process(self, input_data):
        input_data = process(input_data)
        return input_data