import random
import math

class Neuron:
    def __init__(self):
        self.output = 0
        self.weights = [random.uniform(-1, 1) for _ in range(100)]

    def activate(self, inputs):
        self.output = math.tanh(sum([i * w for i, w in zip(inputs, self.weights)]))

    def update(self, inputs):
        self.output = self.activate(inputs)

class Brain:
    def __init__(self):
        self.neurons = [Neuron() for _ in range(100)]

    def think(self, inputs):
        for neuron in self.neurons:
            neuron.update(inputs)

    def mutate(self):
        for neuron in self.neurons:
            neuron.weights = [w + random.uniform(-0.1, 0.1) for w in neuron.weights]

    def optimize(self):
        best_output = 0
        best_weights = []
        for _ in range(1000):
            self.think([random.uniform(-1, 1) for _ in range(100)])
            if self.neurons[0].output > best_output:
                best_output = self.neurons[0].output
                best_weights = [n.weights for n in self.neurons]
        for i, neuron in enumerate(self.neurons):
            neuron.weights = best_weights[i]

brain = Brain()
brain.optimize()