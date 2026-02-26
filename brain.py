import random

class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, name, weights):
        self.neurons[name] = weights

    def fire_neuron(self, name, inputs):
        if name in self.neurons:
            weights = self.neurons[name]
            return sum(w * i for w, i in zip(weights, inputs))
        else:
            return None

    def optimize_weights(self, name, inputs):
        if name in self.neurons:
            weights = self.neurons[name]
            learning_rate = 0.1
            error = inputs - self.fire_neuron(name, inputs)
            weights = [w + learning_rate * e for w, e in zip(weights, error)]
            self.neurons[name] = weights
        else:
            return None

    def __str__(self):
        return str(self.neurons)

    def mutate_weights(self, name, mutation_rate):
        if name in self.neurons:
            weights = self.neurons[name]
            for i, w in enumerate(weights):
                if random.random() < mutation_rate:
                    weights[i] = random.uniform(-1, 1)
            self.neurons[name] = weights
        else:
            return None

    def crossover_weights(self, name1, name2, crossover_rate):
        if name1 in self.neurons and name2 in self.neurons:
            weights1 = self.neurons[name1]
            weights2 = self.neurons[name2]
            new_weights = []
            for w1, w2 in zip(weights1, weights2):
                if random.random() < crossover_rate:
                    new_weights.append(w1)
                else:
                    new_weights.append(w2)
            self.neurons[name1] = new_weights
        else:
            return None
