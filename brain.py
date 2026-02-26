import numpy as np

class Brain:
    def __init__(self):
        self.neurons = np.zeros((1000, 1000), dtype=int)

    def propagate(self, input_vector):
        output_vector = np.zeros((1000,), dtype=int)
        for i in range(1000):
            for j in range(1000):
                output_vector[i] += self.neurons[i, j] * input_vector[j]
        return output_vector

    def train(self, input_vector, target_vector):
        output_vector = self.propagate(input_vector)
        error = np.sum((output_vector - target_vector) ** 2)
        self.neurons += error * np.outer(input_vector, target_vector)