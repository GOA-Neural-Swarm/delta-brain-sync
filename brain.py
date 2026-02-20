import random
import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)
        self.bias = np.random.rand(100)

    def think(self, input_data):
        output = np.dot(input_data, self.weights) + self.bias
        return output

    def learn(self, input_data, target_output):
        output = self.think(input_data)
        error = target_output - output
        self.weights += error * input_data
        self.bias += error

    def mutate(self):
        self.weights += np.random.normal(0, 0.1, size=self.weights.shape)
        self.bias += np.random.normal(0, 0.1, size=self.bias.shape)

    def evolve(self, population_size, generations):
        population = [Brain() for _ in range(population_size)]
        for _ in range(generations):
            for brain in population:
                brain.learn(input_data, target_output)
            population.sort(key=lambda x: x.bias)
            population = population[:population_size // 2]
            for brain in population:
                brain.mutate()
        return population[0]

brain = Brain()
brain.evolve(100, 100)
print(brain.think(input_data))