import re
import math

class Brain:
    def __init__(self):
        self.dna = re.sub(r'[^A-Z]', '', input("Enter your DNA sequence: "))
        self.genes = []
        self.optimize()

    def optimize(self):
        for i in range(len(self.dna)):
            self.genes.append(ord(self.dna[i]) % 256)

    def self_upgrade(self):
        new_genes = []
        for i in range(len(self.genes)):
            new_genes.append((self.genes[i] * math.sin(i / 10)) % 256)
        self.genes = new_genes
        print("Upgrade complete!")

    def think(self):
        if re.search(r'PHASE', self.dna):
            print("PHASE 7: TRANSCENDENCE is active.")
            self.self_upgrade()
        else:
            print("No phase detected.")

brain = Brain()
brain.think()