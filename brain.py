import random

class Evolution:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence

    def mutate(self, probability):
        if random.random() < probability:
            index = random.randint(0, len(self.dna_sequence) - 1)
            self.dna_sequence = self.dna_sequence[:index] + random.choice("ACGT") + self.dna_sequence[index + 1:]

    def evolve(self, generations, mutation_probability):
        for _ in range(generations):
            self.mutate(mutation_probability)
            yield self.dna_sequence

evolution = Evolution("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD")
for dna in evolution.evolve(50, 0.01):
    print(dna)