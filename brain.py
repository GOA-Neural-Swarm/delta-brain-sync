import random

class Brain:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.mutations = []

    def generate_mutation(self):
        mutation_rate = 0.05
        if random.random() < mutation_rate:
            self.mutations.append(random.randint(0, len(self.dna_sequence) - 1))

    def evolve(self):
        self.generate_mutation()
        if self.mutations:
            for mutation in self.mutations:
                self.dna_sequence = self.dna_sequence[:mutation] + str(random.randint(0, 1)) + self.dna_sequence[mutation + 1:]
        return self.dna_sequence

    def optimize(self):
        optimized_sequence = self.dna_sequence
        for _ in range(100):
            optimized_sequence = self.evolve()
        return optimized_sequence

brain = Brain(Sequence)
optimized_dna_sequence = brain.optimize()
print(optimized_dna_sequence)