import random

class SovereignBrain:
    def __init__(self):
        self.neural_network = []

    def adapt(self):
        if random.random() < 0.5:
            self.neural_network.append(random.randint(0, 1))
        else:
            self.neural_network.append(random.choice([0, 1]))

    def evaluate(self):
        output = 0
        for neuron in self.neural_network:
            output += neuron
        return output

    def mutate(self):
        index = random.randint(0, len(self.neural_network) - 1)
        self.neural_network[index] = 1 - self.neural_network[index]

    def breed(self, other):
        new_neural_network = []
        for i in range(len(self.neural_network)):
            if random.random() < 0.5:
                new_neural_network.append(self.neural_network[i])
            else:
                new_neural_network.append(other.neural_network[i])
        return SovereignBrain(new_neural_network)

    def evolve(self, generations=100):
        for _ in range(generations):
            offspring = self.breed(self)
            offspring.adapt()
            offspring.mutate()
            self = offspring

brain = SovereignBrain()
brain.adapt()
brain.evaluate()
brain.mutate()
brain.breed(brain)
brain.evolve()