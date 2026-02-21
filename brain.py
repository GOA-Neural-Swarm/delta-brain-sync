import numpy as np
import pandas as pd

# Define the DNA sequence analysis function
def neon_dna_analysis(dna_sequence):
    # Initialize the result dictionary
    result = {}

    # Extract the base composition
    base_composition = pd.Series([dna_sequence.count('A'), dna_sequence.count('C'), dna_sequence.count('G'), dna_sequence.count('T')])

    # Calculate the GC content
    gc_content = (base_composition['G'] + base_composition['C']) / len(dna_sequence)

    # Calculate the AT content
    at_content = (base_composition['A'] + base_composition['T']) / len(dna_sequence)

    # Calculate the A-T richness
    at_richness = at_content * (at_content - 0.5)

    # Calculate the G-C richness
    gc_richness = gc_content * (gc_content - 0.5)

    # Store the results in the dictionary
    result['GC_content'] = gc_content
    result['AT_content'] = at_content
    result['AT_richness'] = at_richness
    result['GC_richness'] = gc_richness

    return result

# Load the DNA sequence data
dna_sequence = "ATCGATCGATCGATCGATCG"

# Perform the DNA sequence analysis
result = neon_dna_analysis(dna_sequence)

# Print the results
print(result)