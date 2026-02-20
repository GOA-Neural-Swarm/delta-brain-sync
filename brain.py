import re
import random

class NeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        return sum(inputs[i] * self.weights[i] for i in range(len(inputs))) + self.bias

    def propagate(self, inputs):
        return self.activate(inputs)

class EvolveNeuralNetwork(NeuralNetwork):
    def __init__(self, weights, bias, mutation_rate=0.1):
        super().__init__(weights, bias)
        self.mutation_rate = mutation_rate

    def mutate(self):
        for i in range(len(self.weights)):
            if random.random() < self.mutation_rate:
                self.weights[i] += random.uniform(-1, 1)
        self.bias += random.uniform(-1, 1)

    def evolve(self, inputs, outputs):
        error = sum((self.propagate(inputs) - outputs[i]) ** 2 for i in range(len(outputs))) / len(outputs)
        self.mutation_rate *= 0.99
        if error < 0.01:
            return
        self.mutate()
        self.evolve(inputs, outputs)

def optimize_brain(sequences, inputs, outputs):
    weights = [random.uniform(-1, 1) for _ in range(len(inputs))]
    bias = random.uniform(-1, 1)
    network = EvolveNeuralNetwork(weights, bias)
    for _ in range(1000):
        network.evolve(inputs, outputs)
    return network.weights, network.bias

weights, bias = optimize_brain([re.sub('A', '1', re.sub('T', '0', s)) for s in sequences], inputs=[0, 1], outputs=[1, 0])

print(f"Weights: {weights}, Bias: {bias}")