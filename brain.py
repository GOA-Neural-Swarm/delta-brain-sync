import numpy as np
import random

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = np.random.rand(len(sequence), len(sequence))

    def mutate(self):
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence)):
                if random.random() < 0.1:
                    self.weights[i][j] += random.uniform(-0.1, 0.1)

    def crossover(self, other):
        offspring = Brain("")
        for i in range(len(self.sequence)):
            if random.random() < 0.5:
                offspring.sequence += self.sequence[i]
            else:
                offspring.sequence += other.sequence[i]
        return offspring

    def evolve(self, iterations):
        for _ in range(iterations):
            offspring = self.crossover(self)
            offspring.mutate()
            self = offspring

    def predict(self, input):
        output = 0
        for i in range(len(self.sequence)):
            output += self.weights[i][input]
        return output

brain = Brain(sequence)
brain.evolve(100)
print(brain.predict(5))