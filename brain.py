import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)

    def think(self, input_seq):
        hidden_layer = np.dot(input_seq, self.weights)
        output_layer = np.tanh(hidden_layer)
        return output_layer

brain = Brain()
input_seq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
output = brain.think(input_seq)
print(output)