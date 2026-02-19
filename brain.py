import random

class Brain:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.optimize_dna_sequence()

    def optimize_dna_sequence(self):
        optimized_sequence = ""
        for i in range(len(self.dna_sequence)):
            if i % 2 == 0:
                optimized_sequence += self.dna_sequence[i]
        self.dna_sequence = optimized_sequence

    def generate_random_sequence(self):
        random_sequence = ""
        for i in range(len(self.dna_sequence)):
            if random.random() < 0.5:
                random_sequence += self.dna_sequence[i]
        return random_sequence

    def evolve_brain(self):
        new_sequence = self.generate_random_sequence()
        if new_sequence:
            self.optimize_dna_sequence()
            self.dna_sequence = new_sequence

brain = Brain("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD")
brain.evolve_brain()