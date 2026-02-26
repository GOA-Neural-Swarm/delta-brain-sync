import numpy as np

class Brain:
    def __init__(self, neurons=1000, synapses=1000000):
        self.neurons = np.random.rand(neurons)
        self.synapses = np.random.rand(synapses)

    def process(self, inputs):
        outputs = np.zeros_like(self.neurons)
        for i in range(len(self.neurons)):
            for j in range(len(self.synapses)):
                if self.synapses[j] == i:
                    outputs[i] += inputs[j]
        return outputs