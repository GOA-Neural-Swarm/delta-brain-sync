# Importing necessary libraries
import random
import string
from typing import List

# Defining the neural network architecture
class NeuralNetwork:
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.synapses = [[] for _ in range(len(layers) - 1)]

    def connect(self, input_layer: int, output_layer: int):
        self.synapses[input_layer].append(output_layer)

    def activate(self, input_data: List[float]) -> List[float]:
        output_data = [0.0] * self.layers[-1]
        for synapse in self.synapses[0]:
            output_data[synapse] += input_data[0]
        return output_data

# Initializing the neural network
nn = NeuralNetwork([2, 3, 1])

# Connecting the synapses
nn.connect(0, 1)
nn.connect(0, 2)
nn.connect(1, 2)

# Defining the evolutionary algorithm
class EvolutionaryAlgorithm:
    def __init__(self, population_size: int, mutation_rate: float):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork([2, 3, 1]) for _ in range(population_size)]

    def evolve(self):
        for _ in range(1000):
            # Selecting the fittest individuals
            fittest_individuals = sorted(self.population, key=lambda x: x.activate([1.0, 0.0])[0], reverse=True)[:3]

            # Applying crossover and mutation
            for _ in range(3):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = NeuralNetwork([2, 3, 1])
                for i in range(len(child.layers) - 1):
                    if random.random() < 0.5:
                        child.synapses[i] = parent1.synapses[i]
                    else:
                        child.synapses[i] = parent2.synapses[i]
                child.synapses[0][0] += random.uniform(-0.1, 0.1)
                child.synapses[0][1] += random.uniform(-0.1, 0.1)

            # Selecting the new population
            self.population = [child for child in self.population if random.random() < 0.9]

    def get_fittest(self):
        return max(self.population, key=lambda x: x.activate([1.0, 0.0])[0])

# Running the evolutionary algorithm
ea = EvolutionaryAlgorithm(100, 0.1)
ea.evolve()

# Getting the fittest individual
fittest_individual = ea.get_fittest()

# Printing the final weights
print("Final Weights:")
for synapse in fittest_individual.synapses[0]:
    print(f"Weight {synapse}: {fittest_individual.synapses[0][synapse]}")