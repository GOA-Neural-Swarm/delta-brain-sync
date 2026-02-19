import random
import math

class TelefoxX:
    def __init__(self):
        self.memory = []
        self.limits = {"A": 34, "C": 41, "G": 38, "T": 33}

    def evolve(self):
        new_memory = []
        for _ in range(100):
            gene = ""
            for _ in range(5):
                nucleotide = random.choice(list(self.limits.keys()))
                gene += nucleotide
            new_memory.append(gene)
        self.memory = new_memory

    def calculate_fitness(self):
        fitness = 0
        for gene in self.memory:
            score = 0
            for nucleotide in gene:
                score += self.limits[nucleotide]
            fitness += score
        return fitness

    def mutate(self):
        for _ in range(10):
            gene = random.choice(self.memory)
            position = random.randint(0, len(gene) - 1)
            nucleotide = random.choice(list(self.limits.keys()))
            gene = gene[:position] + nucleotide + gene[position + 1:]
            self.memory.append(gene)

    def run(self):
        while True:
            self.evolve()
            fitness = self.calculate_fitness()
            print(f"Fitness: {fitness}")
            if fitness > 5000:
                break
            self.mutate()

telefox = TelefoxX()
telefox.run()