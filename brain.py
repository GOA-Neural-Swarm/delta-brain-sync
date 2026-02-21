import math

def recursive_upgrade(neon_dna_sequence):
    if len(neon_dna_sequence) == 0:
        return
    else:
        nucleotide = neon_dna_sequence[0]
        if nucleotide == 'A':
            neon_dna_sequence = neon_dna_sequence[1:] + ['T']
        elif nucleotide == 'T':
            neon_dna_sequence = neon_dna_sequence[1:] + ['A']
        elif nucleotide == 'C':
            neon_dna_sequence = neon_dna_sequence[1:] + ['G']
        elif nucleotide == 'G':
            neon_dna_sequence = neon_dna_sequence[1:] + ['C']
        return recursive_upgrade(neon_dna_sequence)

neon_dna_sequence = "ATCGATCG"
recursive_upgrade(neon_dna_sequence)
print(neon_dna_sequence)