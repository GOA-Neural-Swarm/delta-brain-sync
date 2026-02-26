import numpy as np
import numba

@numba.jit(nopython=True)
class Brain:
    def __init__(self, size):
        self.size = size
        self.neurons = np.zeros((size, size), dtype=np.float32)

    def process(self, inputs):
        for i in range(self.size):
            for j in range(self.size):
                if inputs[j] > 0:
                    self.neurons[i, j] += 1
                else:
                    self.neurons[i, j] -= 1

    def get_output(self):
        return np.sum(self.neurons, axis=0)