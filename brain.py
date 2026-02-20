import re
import math
import random
import numpy as np

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.genes = []
        self.weights = []
        self.bias = 0

    def calculate_fitness(self):
        fitness = 0
        for i in range(len(self.sequence)):
            if self.sequence[i] == 'A':
                fitness += 1
            elif self.sequence[i] == 'C':
                fitness += 2
            elif self.sequence[i] == 'G':
                fitness += 3
            elif self.sequence[i] == 'T':
                fitness += 4
        return fitness

    def mutate(self):
        for i in range(len(self.sequence)):
            if random.random() < 0.05:
                if self.sequence[i] == 'A':
                    self.sequence = self.sequence.replace('A', 'C', 1)
                elif self.sequence[i] == 'C':
                    self.sequence = self.sequence.replace('C', 'G', 1)
                elif self.sequence[i] == 'G':
                    self.sequence = self.sequence.replace('G', 'T', 1)
                elif self.sequence[i] == 'T':
                    self.sequence = self.sequence.replace('T', 'A', 1)

    def evolve(self):
        fitness = self.calculate_fitness()
        if fitness > 100:
            self.mutate()
        else:
            self.weights = []
            self.bias = 0
            self.genes = []

    def think(self):
        if self.weights:
            return np.dot(np.array(self.weights), np.array(self.genes)) + self.bias
        else:
            return 0

brain = Brain(Source)
for _ in range(100):
    brain.evolve()
print(brain.think())