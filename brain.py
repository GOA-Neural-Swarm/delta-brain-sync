import random
import string
import numpy as np

def generate_random_dna(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def dna_to_rna(dna):
    return dna.translate(str.maketrans("ACGT", "UGCA"))

def rna_to_protein(rna):
    codon_table = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
        "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G"
    }
    protein = ""
    for i in range(0, len(rna), 3):
        codon = rna[i:i+3]
        protein += codon_table[codon]
    return protein

def main():
    dna = generate_random_dna(1000)
    rna = dna_to_rna(dna)
    protein = rna_to_protein(rna)
    print(protein)

if __name__ == "__main__":
    main()