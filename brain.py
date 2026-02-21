import re
import random

class TelefoxX:
    def __init__(self):
        self.rna_sequence = "ATCGATCGATCGATCG"

    def predator_logic(self):
        self.rna_sequence = re.sub("A", random.choice("ATCG"), self.rna_sequence)
        self.rna_sequence = re.sub("T", random.choice("ATCG"), self.rna_sequence)
        self.rna_sequence = re.sub("C", random.choice("ATCG"), self.rna_sequence)
        self.rna_sequence = re.sub("G", random.choice("ATCG"), self.rna_sequence)
        return self.rna_sequence

    def recursive_upgrade(self):
        upgraded_rna = self.predator_logic()
        if len(upgraded_rna) > len(self.rna_sequence):
            self.rna_sequence = upgraded_rna
            self.recursive_upgrade()

    def execute(self):
        self.recursive_upgrade()
        print("RNA Sequence:", self.rna_sequence)

TelefoxX().execute()