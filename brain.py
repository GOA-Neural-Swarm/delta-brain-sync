import numpy as np
import matplotlib.pyplot as plt

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = np.random.rand(len(sequence), len(sequence))

    def evolve(self):
        new_weights = np.zeros((len(self.sequence), len(self.sequence)))
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence)):
                new_weights[i][j] = np.sum(self.sequence[i] * self.sequence[j])
        self.weights = new_weights

    def think(self):
        output = np.zeros(len(self.sequence))
        for i in range(len(self.sequence)):
            output[i] = np.sum(self.weights[i] * self.sequence)
        return output

brain = Brain(PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)
brain.evolve()
print(brain.think())

# Plot the brain activity
plt.plot(brain.think())
plt.xlabel('Neuron Index')
plt.ylabel('Activity Level')
plt.title('Brain Activity Plot')
plt.show()