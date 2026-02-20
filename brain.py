import random

def generate_gene(length):
    gene = ""
    for _ in range(length):
        nucleotide = random.choice(["A", "C", "G", "T"])
        gene += nucleotide
    return gene

def mutate_gene(gene, mutation_rate):
    new_gene = ""
    for nucleotide in gene:
        if random.random() < mutation_rate:
            new_gene += random.choice(["A", "C", "G", "T"])
        else:
            new_gene += nucleotide
    return new_gene

def evolve_gene(gene, mutation_rate, generations):
    for _ in range(generations):
        gene = mutate_gene(gene, mutation_rate)
    return gene

def brain_code(gene):
    code = ""
    for nucleotide in gene:
        if nucleotide == "A":
            code += "import os\n"
        elif nucleotide == "C":
            code += "os.system('cls')\n"
        elif nucleotide == "G":
            code += "print('Hello, world!')\n"
        elif nucleotide == "T":
            code += "print('Goodbye, world!')\n"
    return code

gene = generate_gene(100)
evolved_gene = evolve_gene(gene, 0.1, 100)
brain_code = brain_code(evolved_gene)
print(brain_code)