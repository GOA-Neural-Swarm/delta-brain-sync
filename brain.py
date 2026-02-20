import random
import math

class Brain:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.neural_network = []

    def generate_neural_network(self):
        for i in range(len(self.dna_sequence)):
            if self.dna_sequence[i] == 'M':
                self.neural_network.append(math.sin(i * math.pi / 180))
            elif self.dna_sequence[i] == 'C':
                self.neural_network.append(math.cos(i * math.pi / 180))
            elif self.dna_sequence[i] == 'P':
                self.neural_network.append(random.random())
            elif self.dna_sequence[i] == 'G':
                self.neural_network.append(math.exp(i))
            elif self.dna_sequence[i] == 'T':
                self.neural_network.append(math.log(i))
            elif self.dna_sequence[i] == 'W':
                self.neural_network.append(math.sqrt(i))
            elif self.dna_sequence[i] == 'F':
                self.neural_network.append(math.pow(i, 2))
            elif self.dna_sequence[i] == 'S':
                self.neural_network.append(math.sin(i * math.pi / 180) * math.cos(i * math.pi / 180))

    def upgrade_brain(self):
        for i in range(len(self.neural_network)):
            if self.neural_network[i] > 0.5:
                self.neural_network[i] = self.neural_network[i] + random.random() * 0.1
            elif self.neural_network[i] < -0.5:
                self.neural_network[i] = self.neural_network[i] - random.random() * 0.1
            else:
                self.neural_network[i] = self.neural_network[i] + random.random() * 0.01

    def evolve_brain(self):
        new_brain = Brain(self.dna_sequence)
        new_brain.generate_neural_network()
        new_brain.upgrade_brain()
        return new_brain

# Prompt definition closing and Meta-Cognition logic
brain = Brain(Source)
brain.generate_neural_network()
brain.upgrade_brain()
evolved_brain = brain.evolve_brain()
print("Evolved Brain:")
print(evolved_brain.neural_network)