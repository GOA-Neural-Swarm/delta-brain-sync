import numpy as np

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = np.zeros((neurons, neurons), dtype=int)
        self.synapses = np.zeros((synapses, synapses), dtype=int)

    def process(self, inputs):
        outputs = np.zeros((self.neurons.shape[0],))
        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[0]):
                self.neurons[i][j] += inputs[j]
                if self.neurons[i][j] > 1:
                    outputs[i] = 1
        return outputs

    def learn(self, inputs, outputs):
        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[0]):
                self.synapses[i][j] += inputs[i] * outputs[j]
                if self.synapses[i][j] > 1:
                    self.synapses[i][j] = 1