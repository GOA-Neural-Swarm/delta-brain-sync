import numba
import numpy as np
from numba import prange

class Brain:
    def __init__(self):
        self.neurons = np.zeros((10000, 10000), dtype=int)

    @prange(10000)
    def process(self):
        for node in prange(10000):
            for connected_node in prange(len(self.neurons[node])):
                if random.random() < 0.5:
                    self.neurons[connected_node][node] = 1
                    self.neurons[node][connected_node] = 1

brain = Brain()
start_time = time.time()
brain.process()
end_time = time.time()
print(f"Processing time: {end_time - start_time:.6f} seconds")