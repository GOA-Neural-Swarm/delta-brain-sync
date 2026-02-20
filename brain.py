import re
from collections import Counter

def optimize_brain(neon_dna):
    # Sequence analysis
    dna_len = len(neon_dna)
    nucleotide_freq = Counter(neon_dna)

    # Generate optimized brain code
    optimized_code = ""
    for i in range(dna_len):
        nucleotide = neon_dna[i]
        if nucleotide_freq[nucleotide] > 0.5:
            optimized_code += nucleotide
        else:
            optimized_code += "X"  # Substitute unknown nucleotides

    return optimized_code

# Example usage
neon_dna_sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"
optimized_code = optimize_brain(neon_dna_sequence)
print(optimized_code)