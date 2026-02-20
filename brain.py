import random
import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)

    def predict(self, inputs):
        return np.dot(inputs, self.weights)

    def mutate(self, rate=0.1):
        for i in range(self.inputs):
            for j in range(self.outputs):
                if random.random() < rate:
                    self.weights[i][j] += random.uniform(-1, 1)

    def evolve(self, population_size=100, generations=100):
        population = [self for _ in range(population_size)]
        for _ in range(generations):
            for individual in population:
                individual.mutate()
            population = [max(population, key=lambda x: x.predict([1]))]
        return max(population, key=lambda x: x.predict([1]))

class Brain:
    def __init__(self):
        self.neural_network = NeuralNetwork(1, 1)

    def think(self):
        self.neural_network.evolve()

    def get_output(self):
        return self.neural_network.predict([1])

brain = Brain()
brain.think()
print(brain.get_output())