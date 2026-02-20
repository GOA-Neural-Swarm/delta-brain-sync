import numpy as np
import random
import math

class Brain:
    def __init__(self):
        self.weights = np.random.rand(5127)

    def evolve(self):
        new_weights = np.zeros_like(self.weights)
        for i in range(5127):
            new_weights[i] = self.weights[i] + np.random.normal(0, 0.1)
        self.weights = new_weights

    def think(self, input_data):
        output = 0
        for i in range(5127):
            output += self.weights[i] * input_data[i]
        return output

brain = Brain()

for _ in range(1000):
    brain.evolve()
    input_data = [random.randint(0, 1) for _ in range(5127)]
    output = brain.think(input_data)
    print("Output:", output)