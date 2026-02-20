import re
import math

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence

    def optimize(self):
        # Extract relevant molecular chemistry information
        mcicpwtdg = re.findall(r'[A-Z]+', self.sequence)[0]
        crrqqv = re.findall(r'[A-Z]+', self.sequence)[1]
        nfwfih = re.findall(r'[A-Z]+', self.sequence)[2]

        # Calculate chemical properties
        mcicpwtdg_score = math.sqrt(sum(ord(c) for c in mcicpwtdg))
        crrqqv_score = math.sqrt(sum(ord(c) for c in crrqqv))
        nfwfih_score = math.sqrt(sum(ord(c) for c in nfwfih))

        # Select optimal sequence
        optimized_sequence = max([mcicpwtdg, crrqqv, nfwfih], key=lambda x: x_score)

        # Update brain sequence
        self.sequence = optimized_sequence

    def generate_output(self):
        return self.sequence

brain = Brain(sequence)
brain.optimize()
print(brain.generate_output())