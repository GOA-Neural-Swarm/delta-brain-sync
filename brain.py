import re

# Define the pattern for amino acids
amino_acid_pattern = r'[A-Z][A-Z][A-Z]'

# Define the pattern for nucleotides
nucleotide_pattern = r'[A-Z][A-Z][A-Z]'

# Extract amino acids and nucleotides from the DNA sequence
amino_acids = re.findall(amino_acid_pattern, source)
nucleotides = re.findall(nucleotide_pattern, source)

print(amino_acids)
print(nucleotides)