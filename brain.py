import numpy as np

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons
        self.weights = np.random.rand(neurons, neurons)

    def propagate(self, inputs):
        outputs = np.dot(inputs, self.weights)
        return outputs

    def learn(self, inputs, targets):
        outputs = self.propagate(inputs)
        errors = targets - outputs
        self.weights += np.dot(inputs.T, errors)
        return outputs, errors