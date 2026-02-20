import random

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.genome = ''.join(random.sample(sequence, len(sequence)))

    def mutate(self):
        for i in range(len(self.sequence)):
            if random.random() < 0.1:
                self.sequence = self.sequence[:i] + chr(ord(self.sequence[i]) + random.randint(-1, 1)) + self.sequence[i+1:]
                self.genome = self.genome[:i] + chr(ord(self.genome[i]) + random.randint(-1, 1)) + self.genome[i+1:]

    def evaluate(self):
        fitness = 0
        for i in range(len(self.sequence)):
            if self.sequence[i] == self.genome[i]:
                fitness += 1
        return fitness / len(self.sequence)

    def evolve(self, generations):
        for _ in range(generations):
            self.mutate()
            self.genome = self.genome[:int(len(self.genome) * 0.1)] + self.genome[int(len(self.genome) * 0.1):]

    def print_sequence(self):
        print(self.sequence)

brain = Brain(Source)
brain.evolve(100)
brain.print_sequence()