import random
import numpy as np

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.fitness = 0

    def calculate_fitness(self):
        fitness = 0
        for i in range(len(self.sequence) - 1):
            if self.sequence[i] == self.sequence[i+1]:
                fitness += 1
        self.fitness = fitness

    def mutate(self):
        index = random.randint(0, len(self.sequence) - 1)
        self.sequence = list(self.sequence)
        self.sequence[index] = random.choice('ATCG')
        self.sequence = ''.join(self.sequence)

    def evolve(self, population_size):
        population = [Brain(self.sequence) for _ in range(population_size)]
        for _ in range(100):  # Evolution for 100 generations
            population.sort(key=lambda x: x.fitness, reverse=True)
            offspring = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(population[:population_size//2], 2)
                child = Brain(''.join([random.choice(parent1.sequence), random.choice(parent2.sequence)]))
                child.mutate()
                offspring.append(child)
            population = offspring
        self.fitness = population[0].fitness

brain = Brain('MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD')
brain.evolve(100)
print(brain.fitness)