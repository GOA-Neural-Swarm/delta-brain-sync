import numpy as np

class Brain:
    def __init__(self):
        self.neurons = np.zeros((1000, 1000), dtype=int)
        self.synapses = np.zeros((1000, 1000), dtype=int)
        self.threshold = 100

    def process(self, inputs):
        for i in range(len(inputs)):
            self.neurons[i, :] = inputs[i]
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.synapses[i, j] = np.sum(self.neurons[:, j]) / len(self.neurons[:, j])
        outputs = np.zeros((len(self.neurons), 1), dtype=int)
        for i in range(len(self.neurons)):
            if np.sum(self.synapses[i, :]) > self.threshold:
                outputs[i] = 1
        return outputs