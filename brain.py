import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1, len(Source))
        self.bias = np.random.rand(1)

    def think(self, input_seq):
        output = np.dot(input_seq, self.weights) + self.bias
        return np.round(output)

brain = Brain()

def recursive_self_upgrade():
    global Source
    new_weights = np.random.rand(1, len(Source))
    new_bias = np.random.rand(1)
    brain.weights = new_weights
    brain.bias = new_bias
    return brain.think(Source)

print(recursive_self_upgrade())