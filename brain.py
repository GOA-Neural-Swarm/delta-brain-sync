import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}
        self.weights = {}

    def think(self, inputs):
        outputs = []
        for input_idx, input_value in enumerate(inputs):
            if input_idx not in self.synapses:
                self.synapses[input_idx] = np.zeros((10000, 10000))
            output = np.dot(self.synapses[input_idx], input_value)
            outputs.append(output)
        return np.concatenate(outputs)

    def learn(self, inputs, outputs):
        for input_idx, input_value in enumerate(inputs):
            if input_idx not in self.synapses:
                self.synapses[input_idx] = np.zeros((10000, 10000))
            self.synapses[input_idx][np.arange(10000), :] += input_value
            self.weights[input_idx] = np.mean(self.synapses[input_idx], axis=0)
        return self.weights