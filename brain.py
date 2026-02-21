import random
import numpy as np

class DNA:
    def __init__(self, seq):
        self.seq = seq
        self.rna = ''.join([c.upper() if c.isupper() else c.lower() for c in self.seq])

    def mutate(self):
        indices = np.random.choice(range(len(self.rna)), replace=False, size=5)
        for i in indices:
            self.rna = self.rna[:i] + random.choice(['A', 'C', 'G', 'T']) + self.rna[i+1:]

    def transcribe(self):
        return self.rna

    def predict_next_base(self):
        probabilities = {'A': 0.3, 'C': 0.2, 'G': 0.3, 'T': 0.2}
        next_base = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
        return next_base

# Generate a random DNA sequence
dna = DNA(''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(100)]))

# Perform 5 iterations of mutation, transcription, and prediction
for _ in range(5):
    dna.mutate()
    print(dna.transcribe())
    print(dna.predict_next_base())