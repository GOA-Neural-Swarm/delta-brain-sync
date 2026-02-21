import random
import math

class Predator:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.brain_logic = self.generate_brain_logic()

    def generate_brain_logic(self):
        logic = ""
        for i in range(len(self.dna_sequence)):
            if self.dna_sequence[i] == "A":
                logic += "A" + str(random.randint(0, 1))
            elif self.dna_sequence[i] == "C":
                logic += "C" + str(random.randint(0, 1))
            elif self.dna_sequence[i] == "G":
                logic += "G" + str(random.randint(0, 1))
            elif self.dna_sequence[i] == "T":
                logic += "T" + str(random.randint(0, 1))
        return logic

    def execute_brain_logic(self):
        for i in range(len(self.brain_logic)):
            if self.brain_logic[i] == "A0":
                # Perform action A0
                print("Action A0 executed")
            elif self.brain_logic[i] == "C1":
                # Perform action C1
                print("Action C1 executed")
            elif self.brain_logic[i] == "G0":
                # Perform action G0
                print("Action G0 executed")
            elif self.brain_logic[i] == "T1":
                # Perform action T1
                print("Action T1 executed")

# Example usage
dna_sequence = "ATCGATCG"
predator = Predator(dna_sequence)
predator.execute_brain_logic()