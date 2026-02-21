import json
import random
from deap import base, creator, tools, algorithms

# Load neural network weights
weights = json.load(open('neural_weights.json', 'r'))

# Define the fitness function
def fitness(individual):
    return sum(individual),

# Create the fitness class
creator.create('FitnessMin', base.Fitness, weights=fitness)

# Create the individual class
creator.create('Individual', list, fitness=creator.FitnessMin)

# Create the toolbox
toolbox = base.Toolbox()

# Register the genetic operator
toolbox.register('attr_bool', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n=1000)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Create a population of 1000 individuals
pop = toolbox.population(n=1000)

# Evaluate the population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = (fit,)

# Perform a single generation of the genetic algorithm
pop, stats = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=10, verbose=True)

# Get the best individual
best_individual = tools.selBest(pop, k=1)[0]

# Print the best individual
print('Best Individual:', best_individual)

# Print the best fitness
print('Best Fitness:', best_individual.fitness.values[0])