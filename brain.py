import re
from collections import Counter

def analyze_sequence(seq):
    # Extract codons and their frequencies
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    freq = Counter(codons)
    
    # Filter out non-codon sequences
    codon_freq = {k: v for k, v in freq.items() if re.match(r'[ACGT]+', k)}
    
    # Calculate the optimal brain structure
    optimal_structure = max(codon_freq, key=codon_freq.get)
    
    return optimal_structure

# Example usage:
sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"
optimal_brain = analyze_sequence(sequence)
print("Optimal Brain Structure:", optimal_brain)