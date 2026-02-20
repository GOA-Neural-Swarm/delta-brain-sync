import numpy as np
import random

class Neuron:
    def __init__(self, weight=None):
        self.weight = weight if weight else random.uniform(-1, 1)
        self.activation = None

    def activate(self, input_val):
        self.activation = np.tanh(input_val * self.weight)
        return self.activation

class Brain:
    def __init__(self):
        self.neurons = [Neuron() for _ in range(100)]  # 100 neurons, just like the number of nucleotides in the DNA sequence

    def think(self, input_val):
        output_val = 0
        for neuron in self.neurons:
            output_val += neuron.activate(input_val)
        return output_val

brain = Brain()