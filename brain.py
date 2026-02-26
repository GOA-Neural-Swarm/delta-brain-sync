import numpy as np
import numba

@numba.jit(nopython=True)
class Brain:
    def __init__(self):
        self.synaptic_weights = np.random.rand(10, 10)

    def think(self, inputs):
        return np.dot(inputs, self.synaptic_weights)