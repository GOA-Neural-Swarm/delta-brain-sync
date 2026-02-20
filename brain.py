import numpy as np
import random
import copy

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.mutation_rate = 0.01
        self.selection_pressure = 0.5

    def generate_child(self):
        child_sequence = copy.deepcopy(self.sequence)
        for i in range(len(child_sequence)):
            if random.random() < self.mutation_rate:
                child_sequence[i] = random.choice('ACGT')
        return child_sequence

    def evaluate_fitness(self, sequence):
        fitness = 0
        for i in range(len(sequence)):
            if sequence[i] in 'ACGT':
                fitness += 1
            else:
                fitness -= 1
        return fitness

    def select_parent(self, population):
        parents = []
        while len(parents) < 2:
            population_fitness = [self.evaluate_fitness(seq) for seq in population]
            max_fitness_idx = np.argmax(population_fitness)
            parents.append(population[max_fitness_idx])
            population.pop(max_fitness_idx)
        return parents

    def evolve(self, population):
        parents = self.select_parent(population)
        child1 = self.generate_child()
        child2 = self.generate_child()
        population.append(child1)
        population.append(child2)

    def optimize(self, iterations):
        population = [self.sequence] * 10
        for _ in range(iterations):
            self.evolve(population)
        best_sequence = max(population, key=self.evaluate_fitness)
        return best_sequence

brain = Brain(Source)
optimized_sequence = brain.optimize(100)
print(optimized_sequence)