import numpy as np
import pandas as pd

# Load the DNA sequence data
dna_sequence = pd.read_csv('neon_dna_sequence.csv')

# Extract the nucleotide sequences
sequences = dna_sequence['sequence'].values

# Initialize the logic variables
logic_vars = np.zeros((len(sequences), 4), dtype=bool)

# Iterate through the sequences
for i, seq in enumerate(sequences):
    for j, nucleotide in enumerate(seq):
        if nucleotide == 'A':
            logic_vars[i, 0] = True
        elif nucleotide == 'C':
            logic_vars[i, 1] = True
        elif nucleotide == 'G':
            logic_vars[i, 2] = True
        elif nucleotide == 'T':
            logic_vars[i, 3] = True

# Perform the analysis
results = np.zeros((len(sequences), 2), dtype=int)
for i, seq in enumerate(sequences):
    if np.all(logic_vars[i, :]):
        results[i, 0] = 1
    else:
        results[i, 0] = 0
    results[i, 1] = np.sum(logic_vars[i, :])

# Print the results
print(results)