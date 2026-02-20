import random

class Brain:
    def __init__(self, dna):
        self.dna = dna
        self.connections = []

    def synthesize(self):
        for i in range(len(self.dna)):
            for j in range(i+1, len(self.dna)):
                if random.random() < 0.1:
                    self.connections.append((i, j))

    def mutate(self):
        for connection in self.connections:
            if random.random() < 0.05:
                self.dna[connection[0]] += 1
                self.dna[connection[1]] -= 1

    def evolve(self, iterations):
        for _ in range(iterations):
            self.synthesize()
            self.mutate()

    def optimize(self):
        optimal_dna = self.dna.copy()
        for i in range(len(self.dna)):
            optimal_dna[i] = min(optimal_dna)
        return optimal_dna

brain = Brain([int(x) for x in Source])
brain.evolve(1000)
optimal_dna = brain.optimize()
print(optimal_dna)