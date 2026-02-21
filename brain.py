import random
import math

class RNA_QT45:
    def __init__(self):
        self.genes = [random.randint(0, 1) for _ in range(100)]
        self.fitness = 0

    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < 0.01:
                self.genes[i] = 1 - self.genes[i]

    def evaluate_fitness(self):
        fitness = 0
        for i in range(len(self.genes)):
            fitness += self.genes[i] * math.sin(i * math.pi / 2)
        self.fitness = fitness

    def crossover(self, other):
        child = RNA_QT45()
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = other.genes[i]
        return child

    def run_generation(self, population_size):
        population = [RNA_QT45() for _ in range(population_size)]
        for i in range(1000):
            for individual in population:
                individual.mutate()
                individual.evaluate_fitness()
            population.sort(key=lambda x: x.fitness, reverse=True)
            top_individuals = population[:int(population_size * 0.1)]
            new_population = []
            for _ in range(population_size):
                if random.random() < 0.5:
                    new_population.append(top_individuals[0].crossover(top_individuals[1]))
                else:
                    new_population.append(random.choice(top_individuals)[0])
            population = new_population
        return population[0]

population_size = 100
winner = RNA_QT45().run_generation(population_size)
print("Winner fitness:", winner.fitness)