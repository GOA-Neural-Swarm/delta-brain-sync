import random
import math

class RNAQT45:
    def __init__(self, init_population_size=1000, mutation_rate=0.01, crossover_rate=0.5):
        self.population_size = init_population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [random.randint(0, 255) for _ in range(self.population_size)]

    def evolve(self, generations=100):
        for _ in range(generations):
            self.population = self.select_fittest(self.population)
            self.population = self.mutate(self.population, self.mutation_rate)
            self.population = self.crossover(self.population, self.crossover_rate)
        return self.population

    def select_fittest(self, population):
        fittest = max(population, key=lambda x: self.fitness(x))
        return [x for x in population if x == fittest]

    def fitness(self, individual):
        return math.exp(-math.sqrt(sum((x - 128) ** 2 for x in individual)) / len(individual))

    def mutate(self, population, mutation_rate):
        mutated = []
        for individual in population:
            if random.random() < mutation_rate:
                mutated.append([random.randint(0, 255) for _ in range(len(individual))])
            else:
                mutated.append(individual)
        return mutated

    def crossover(self, population, crossover_rate):
        crossovered = []
        for i in range(0, len(population), 2):
            if random.random() < crossover_rate:
                parent1 = population[i]
                parent2 = population[i + 1]
                crossovered.append([x for x in parent1 if random.random() < 0.5])
                crossovered.append([x for x in parent2 if random.random() < 0.5])
            else:
                crossovered.append(parent1)
                crossovered.append(parent2)
        return crossovered

# Initialize and evolve RNA QT45 Predator Logic
rq45 = RNAQT45()
final_population = rq45.evolve(generations=1000)

# Print final population
print([x for x in final_population])