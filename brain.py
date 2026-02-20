import numpy as np

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = np.random.rand(len(sequence), len(sequence))

    def synthesize(self):
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence)):
                self.weights[i][j] = np.exp(-((self.sequence[i] - self.sequence[j]) ** 2) / (2 * 0.1 ** 2))

    def simulate(self):
        output = np.zeros(len(self.sequence))
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence)):
                output[i] += self.weights[i][j] * self.sequence[j]
        return output

brain = Brain(sequence)
brain.synthesize()
output = brain.simulate()
print(output)