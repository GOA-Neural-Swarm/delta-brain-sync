import numpy as np

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons
        self.synapses = np.random.rand(neurons, neurons)
        self.activations = np.zeros(neurons)

    def process(self, inputs):
        for i in range(self.neurons):
            self.activations[i] = np.sum(inputs * self.synapses[:, i]) + self.activations[i]
        return np.tanh(self.activations)

    def learn(self, inputs, targets):
        errors = np.sum((targets - self.process(inputs)) ** 2)
        self.synapses += np.outer(inputs, self.process(inputs) - targets) / errors