import random

def generate_synthetic_dna(length):
    synthetic_dna = ""
    for _ in range(length):
        if random.random() < 0.5:
            synthetic_dna += "A"
        else:
            synthetic_dna += "C"
    return synthetic_dna

synthetic_dna = generate_synthetic_dna(1000)
print(synthetic_dna)