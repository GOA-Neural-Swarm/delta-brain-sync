import numpy as np

class Brain:
    def __init__(self):
        self.neurons = np.random.rand(1000, 1000)

    def process(self, inputs):
        outputs = np.dot(inputs, self.neurons)
        return outputs

    def update(self, inputs, outputs):
        self.neurons += 0.01 * np.dot(np.transpose(inputs), outputs)