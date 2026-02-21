import math
import random
from deap import base, creator, tools, algorithms

# Define the fitness function
def fitness(individual):
    return sum(individual),

# Create the creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create the toolbox
toolbox = base.Toolbox()

# Register the genetic operator
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Create a population of 50 individuals
pop = toolbox.population(n=50)

# Run the genetic algorithm
best_individual = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, verbose=True)
print("Best individual:", best_individual)

# Evaluate the best individual
fitness(best_individual)

# Print the fitness value
print("Fitness value:", fitness(best_individual))