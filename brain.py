import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(1000, 1000)
        self.biases = np.random.rand(1000)

    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.weights) + self.biases
        output_layer = np.maximum(hidden_layer, 0)
        return output_layer

    def backward_pass(self, inputs, targets):
        hidden_layer = self.forward_pass(inputs)
        output_layer = hidden_layer
        error = targets - output_layer
        return error