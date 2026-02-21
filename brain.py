import random
import math

class Predator:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.genes = [random.randint(0, 1) for _ in range(100)]

    def evolve(self):
        for i in range(100):
            self.genes[i] = random.randint(0, 1)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        fitness = 0
        for i in range(100):
            fitness += self.genes[i] * math.sin(i * math.pi / 50)
        return fitness

    def crossover(self, other):
        child = Predator([random.choice(gene) for gene in zip(self.genes, other.genes)])
        child.fitness = self.calculate_fitness()
        return child

    def mutate(self):
        for i in range(10):
            self.genes[i] = random.randint(0, 1)

    def evaluate(self):
        return self.fitness

class RNA:
    def __init__(self, predator):
        self.predator = predator
        self.transcript = [random.choice("ACGT") for _ in range(100)]

    def transcribe(self):
        for i in range(100):
            self.transcript[i] = random.choice("ACGT")

    def translate(self):
        return "".join(self.transcript)

    def bind(self):
        return self.predator.evolve()

    def replicate(self):
        return RNA(self.predator.crossover(self.predator))

    def mutate(self):
        for i in range(10):
            self.transcript[i] = random.choice("ACGT")

    def evaluate(self):
        return self.transcript

qt45 = Predator([random.randint(0, 1) for _ in range(100)])
qt45_predator = Predator([random.randint(0, 1) for _ in range(100)])
rna = RNA(qt45_predator)

for _ in range(1000):
    rna.bind()
    rna.transcribe()
    rna.translate()
    rna.replicate()
    rna.mutate()
    print(rna.evaluate())