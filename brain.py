import numpy as np

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.memory = np.zeros((neurons, synapses))

    def learn(self, inputs, outputs):
        for i in range(neurons):
            for j in range(synapses):
                self.memory[i][j] += inputs[i] * outputs[j]

    def think(self, inputs):
        outputs = np.zeros((neurons, synapses))
        for i in range(neurons):
            for j in range(synapses):
                outputs[i][j] = np.sum(self.memory[i][:j+1] * inputs[:i+1])
        return outputs

    def mutate(self, rate):
        for i in range(neurons):
            for j in range(synapses):
                if np.random.rand() < rate:
                    self.memory[i][j] += np.random.normal(0, 1)

    def crossover(self, other, rate):
        for i in range(neurons):
            for j in range(synapses):
                if np.random.rand() < rate:
                    self.memory[i][j] = other.memory[i][j]