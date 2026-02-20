import numpy as np

class TelefoxX:
    def __init__(self):
        self.dna = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"
        self.brain = np.zeros((len(self.dna), len(self.dna)), dtype=int)

    def mutate(self):
        for i in range(len(self.dna)):
            if np.random.rand() < 0.1:
                self.dna[i] = np.random.choice("ACGT")

    def evolve(self):
        self.brain = np.zeros((len(self.dna), len(self.dna)), dtype=int)
        for i in range(len(self.dna)):
            for j in range(len(self.dna)):
                self.brain[i][j] = np.random.randint(0, 2)

        for _ in range(100):
            self.mutate()
            fitness = self.calculate_fitness()
            self.brain += fitness

    def calculate_fitness(self):
        fitness = 0
        for i in range(len(self.dna)):
            for j in range(len(self.dna)):
                if self.dna[i] == self.dna[j]:
                    fitness += self.brain[i][j]
        return fitness

    def run(self):
        self.evolve()
        print(self.dna)

TelefoxX().run()