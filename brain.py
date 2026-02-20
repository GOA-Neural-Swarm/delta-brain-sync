import random

class Brain:
    def __init__(self, seq):
        self.seq = seq
        self.genes = []

    def generate_genes(self):
        for i in range(len(self.seq) // 2):
            self.genes.append((self.seq[i * 2], self.seq[i * 2 + 1]))

    def mutate_genes(self):
        for i in range(len(self.genes)):
            if random.random() < 0.1:
                self.genes[i] = (random.choice("PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"), self.genes[i][1])
            if random.random() < 0.05:
                self.genes[i] = (self.genes[i][0], random.choice("PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"))

    def evaluate_fitness(self):
        fitness = 0
        for i in range(len(self.genes)):
            if self.genes[i][0] == self.genes[i][1]:
                fitness += 1
        return fitness

    def evolve(self, iterations):
        for _ in range(iterations):
            self.generate_genes()
            self.mutate_genes()
            fitness = self.evaluate_fitness()
            if fitness > self.genes[0][0]:
                self.genes = [(self.genes[0][0], self.genes[0][1])]
        return self.genes

brain = Brain("PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW")
evolved_genes = brain.evolve(1000)
print(evolved_genes)