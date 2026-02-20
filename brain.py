import random
import numpy as np
from deap import base, creator, tools, algorithms

# Define the fitness function
def evaluate(individual):
    return sum(individual),

# Create the toolbox
toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the genetic operators
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic algorithm
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)

# Run the genetic algorithm
for _ in range(10):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.1)
    fitness = map(evaluate, offspring)
    for fit, ind in zip(fitness, offspring):
        ind.fitness.values = fit
    hof.update(offspring)

# Print the best individual
print(hof[0])