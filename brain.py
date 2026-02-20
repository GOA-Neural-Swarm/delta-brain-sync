import random

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = [random.uniform(0, 1) for _ in range(len(sequence))]

    def process(self):
        output = 0
        for i in range(len(self.sequence)):
            output += self.weights[i] * self.sequence[i]
        return output

    def optimize(self):
        for i in range(len(self.sequence)):
            self.weights[i] = random.uniform(0, 1)

    def evolve(self):
        new_sequence = [random.choice([0, 1]) for _ in range(len(self.sequence))]
        new_brain = Brain(new_sequence)
        new_brain.optimize()
        return new_brain

brain = Brain(PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)
print(brain.process())
brain.evolve()
print(brain.process())