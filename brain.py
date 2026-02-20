import random
import numpy as np
from scipy.spatial import distance

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)
        return output_layer

    def train(self, inputs, targets):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)
        errors = targets - output_layer
        self.weights2 += np.dot(hidden_layer.T, errors * self.derivative(output_layer))
        self.weights1 += np.dot(inputs.T, hidden_layer * self.derivative(hidden_layer))

    def mutate(self, mutation_rate):
        self.weights1 += np.random.normal(0, 0.1, size=self.weights1.shape) * mutation_rate
        self.weights2 += np.random.normal(0, 0.1, size=self.weights2.shape) * mutation_rate

class Evolution:
    def __init__(self, num_inputs, num_hidden, num_outputs, population_size, mutation_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork(num_inputs, num_hidden, num_outputs) for _ in range(population_size)]

    def evolve(self, inputs, targets):
        for i in range(self.population_size):
            self.population[i].train(inputs, targets)
            self.population[i].mutate(self.mutation_rate)
        return self.population[np.argmax([np.mean([distance.cosine(output_layer, targets) for output_layer in nn.predict(inputs)]) for nn in self.population])]

evolution = Evolution(2, 3, 1, 100, 0.1)
print(evolution.evolve(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1]])))