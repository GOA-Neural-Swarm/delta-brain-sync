import re
import math

class NeuroCore:
    def __init__(self, sequence):
        self.sequence = sequence
        self.gene_map = {}

    def map_genes(self):
        pattern = re.compile(r'([A-Za-z]+)')
        for match in pattern.finditer(self.sequence):
            gene = match.group(1)
            self.gene_map[gene] = self.gene_map.get(gene, 0) + 1

    def optimize_brain(self):
        for gene, frequency in self.gene_map.items():
            if frequency > 1:
                self.sequence = self.sequence.replace(gene, gene + '_' + str(frequency))

    def upgrade(self):
        self.map_genes()
        self.optimize_brain()
        self.sequence = ''.join(sorted(self.sequence))

    def evolve(self):
        new_sequence = ''
        for char in self.sequence:
            if char.isalpha():
                new_sequence += chr((ord(char) + 1) % 26)
            else:
                new_sequence += char
        self.sequence = new_sequence
        self.upgrade()

    def __str__(self):
        return self.sequence

brain = NeuroCore(Source)
brain.evolve()
print(brain)