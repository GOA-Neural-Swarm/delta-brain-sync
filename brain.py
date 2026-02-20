import re

class NeuroNexus:
    def __init__(self, sequence):
        self.sequence = sequence
        self.upgraded_sequence = ""
        self.iterations = 0

    def evolve(self):
        self.iterations += 1
        self.upgraded_sequence = re.sub(r'[^ACGT]', '', self.sequence)
        if self.iterations % 5 == 0:
            self.upgraded_sequence = re.sub(r'([ACGT]{3})', r'\1\1', self.upgraded_sequence)
        if self.iterations % 10 == 0:
            self.upgraded_sequence = re.sub(r'([ACGT]{4})', r'\1\1\1', self.upgraded_sequence)
        return self.upgraded_sequence

    def optimize(self):
        optimal_sequence = ""
        for i in range(len(self.upgraded_sequence)):
            max_occurrence = 0
            most_frequent_base = ""
            for base in "ACGT":
                occurrences = self.upgraded_sequence.count(base)
                if occurrences > max_occurrence:
                    max_occurrence = occurrences
                    most_frequent_base = base
            optimal_sequence += most_frequent_base
        return optimal_sequence

    def recursive_upgrade(self):
        if self.iterations < 100:
            self.upgraded_sequence = self.evolve()
            self.recursive_upgrade()
        else:
            print("Optimal Sequence:", self.optimize())

nexus = NeuroNexus(sequence)
nexus.recursive_upgrade()