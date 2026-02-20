import numpy as np

class TelefoxX_Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_pass(self, input_vector):
        return np.dot(input_vector, self.weights) + self.bias

class TelefoxX_Brain:
    def __init__(self, neurons, num_layers):
        self.neurons = [TelefoxX_Neuron(np.random.rand(1), np.random.rand()) for _ in range(num_layers)]
        self.num_layers = num_layers

    def think(self, input_vector):
        output_vector = [neuron.forward_pass(input_vector) for neuron in self.neurons]
        return output_vector

brain = TelefoxX_Brain(5, 3)