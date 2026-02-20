import random

class TelefoxXBrain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.upgrade_factors = [random.randint(1, 10) for _ in range(len(sequence))]

    def process_sequence(self):
        upgraded_sequence = ""
        for i, char in enumerate(self.sequence):
            if self.upgrade_factors[i] % 2 == 0:
                upgraded_sequence += char.upper()
            else:
                upgraded_sequence += char.lower()
        return upgraded_sequence

    def recursive_upgrade(self, iterations):
        for _ in range(iterations):
            upgraded_sequence = self.process_sequence()
            self.upgrade_factors = [random.randint(1, 10) for _ in range(len(upgraded_sequence))]
            self.sequence = upgraded_sequence
        return self.sequence

brain = TelefoxXBrain(sequence)
print(brain.recursive_upgrade(5))