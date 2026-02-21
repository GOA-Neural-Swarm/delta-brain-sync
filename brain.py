import numpy as np
import matplotlib.pyplot as plt

# Define the DNA sequence
dna_sequence = 'ATCGATCGATCGATCGATCG'

# Define the codon table
codon_table = {
    'ATC': 'A', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAG': 'K', 'AAA': 'K', 'AAT': 'N', 'AGC': 'S', 'AGT': 'S',
    'AGG': 'R', 'AGA': 'R', 'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'AAC': 'N',
    'CAA': 'Q', 'CAG': 'Q', 'CAC': 'H', 'CAA': 'Q', 'CAT': 'H', 'CGC': 'R',
    'CGT': 'R', 'CGG': 'R', 'CGA': 'R', 'CTA': 'L', 'CTC': 'L', 'CTG': 'L',
    'CTT': 'L', 'GAT': 'D', 'GAC': 'D', 'GAG': 'E', 'GAA': 'E', 'GAT': 'D',
    'GCT': 'A', 'GCC': 'A', 'GCG': 'A', 'GCA': 'A', 'GTG': 'V', 'GTC': 'V',
    'GTT': 'V', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GTA': 'V', 'GTC': 'V',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TCA': 'S', 'TCC': 'S',
    'TCG': 'S', 'TCT': 'S', 'TGA': '*', 'TGG': 'W', 'TGT': 'C', 'TGC': 'C',
    'TTA': 'L', 'TTC': 'L', 'TTG': 'L', 'TTT': 'L'
}

# Define the start and stop codons
start_codon = 'ATG'
stop_codon = 'TAA'

# Initialize the amino acid sequence
amino_acid_sequence = ''

# Iterate over the DNA sequence in 3-base codon increments
for i in range(0, len(dna_sequence), 3):
    # Extract the current codon
    codon = dna_sequence[i:i+3]

    # Look up the amino acid corresponding to the current codon
    amino_acid = codon_table[codon]

    # Append the amino acid to the amino acid sequence
    amino_acid_sequence += amino_acid

# Print the amino acid sequence
print(amino_acid_sequence)