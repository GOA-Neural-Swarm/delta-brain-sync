import random
import math
import copy
from deap import base, creator, tools, algorithms

# Define the problem parameters
POP_SIZE = 100
TOURNAMENT_SIZE = 3
GENE_MUTATION_RATE = 0.1
GENE_MUTATION_STEP = 0.1

# Define the problem function
def problem(individual):
    fitness = 0
    for i in range(len(individual)):
        if random.random() < 0.5:
            fitness += 1
    return fitness,

# Create the toolbox
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register the genetic operators
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", problem)

# Register the genetic algorithm
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# Run the genetic algorithm
pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register(lambda ind: ind.fitness.values[0], name="avg")
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats, halloffame=hof)

# Print the results
print("Best individual is", hof[0])
print("Best fitness is", hof[0].fitness.values[0])