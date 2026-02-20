import random

class Brain:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.upgrade_iterations = 0

    def generate_cognition(self):
        cognition = ""
        for nucleotide in self.dna_sequence:
            if nucleotide in ['A', 'T', 'G', 'C']:
                cognition += nucleotide
            elif nucleotide == 'P':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'G':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'R':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'I':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'W':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'M':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'K':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'Q':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'E':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'L':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'F':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'S':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'T':
                cognition += random.choice(['+', '-'])
            elif nucleotide == 'N':
                cognition += random.choice(['+', '-'])
        return cognition

    def upgrade_brain(self):
        self.upgrade_iterations += 1
        if self.upgrade_iterations % 10 == 0:
            self.dna_sequence = self.generate_cognition()
            print(f"Upgrade Iteration {self.upgrade_iterations}: DNA Sequence Updated")

brain = Brain(PCGNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)
brain.upgrade_brain()