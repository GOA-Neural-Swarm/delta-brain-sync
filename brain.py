import numpy as np

class Brain:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_neurons, num_neurons)

    def process(self, inputs):
        return np.dot(inputs, self.weights)