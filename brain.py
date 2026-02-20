import numpy as np
import scipy.stats as stats
import pandas as pd

class Brain:
    def __init__(self, dna_sequence):
        self.dna = dna_sequence
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.brain_activity = []

    def generate_activity(self):
        for nucleotide in self.dna:
            if nucleotide in self.nucleotides:
                activity = stats.norm.rvs(loc=0, scale=1)
                self.brain_activity.append(activity)
            else:
                print("Invalid nucleotide:", nucleotide)

    def process_activity(self):
        self.brain_activity = np.array(self.brain_activity)
        self.brain_activity = self.brain_activity.mean(axis=0)
        return self.brain_activity

    def synthesize_brain(self):
        self.generate_activity()
        brain_activity = self.process_activity()
        return brain_activity

brain = Brain("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD")
brain_synthesized_activity = brain.synthesize_brain()
print(brain_synthesized_activity)