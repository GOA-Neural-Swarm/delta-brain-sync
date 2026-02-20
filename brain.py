import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Initialize population and mutation rates
POPULATION_SIZE = 100
MUTATION_RATE = 0.1

# Define the brain structure
BRAIN_STRUCTURE = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
])

# Define the fitness function
def fitness(brain):
    # Evaluate the brain's performance using the given sequence
    sequence = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"
    predictions = []
    for i in range(len(sequence) - 2):
        input_vector = brain[i:i+3]
        prediction = np.argmax(input_vector)
        predictions.append(prediction)
    accuracy = accuracy_score([ord(c) for c in sequence], predictions)
    return accuracy

# Main evolution loop
for generation in range(100):
    # Initialize the population
    population = np.random.rand(POPULATION_SIZE, len(BRAIN_STRUCTURE))
    population /= population.sum(axis=1, keepdims=True)

    # Evaluate the population's fitness
    fitness_scores = np.zeros(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        fitness_scores[i] = fitness(population[i])

    # Select the fittest individuals
    indices = np.argsort(fitness_scores)[::-1][:int(POPULATION_SIZE * 0.5)]
    population = population[indices]

    # Perform crossover and mutation
    offspring = np.zeros((POPULATION_SIZE, len(BRAIN_STRUCTURE)))
    for i in range(POPULATION_SIZE):
        parent1 = np.random.choice(indices, p=[1/len(indices)]*len(indices))
        parent2 = np.random.choice(indices, p=[1/len(indices)]*len(indices))
        if np.random.rand() < MUTATION_RATE:
            offspring[i] = np.random.rand(len(BRAIN_STRUCTURE))
        else:
            offspring[i] = 0.5 * population[parent1] + 0.5 * population[parent2]

    # Replace the least fit individuals with the offspring
    population = np.concatenate((population, offspring))

    # Plot the evolution of the population's fitness
    plt.plot(fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of the Population's Fitness")
    plt.show()

    # Print the best brain found so far
    best_brain = population[np.argmax(fitness_scores)]
    print("Best Brain:")
    print(best_brain)