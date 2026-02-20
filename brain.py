import numpy as np

class Neuron:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights
        self.output = np.dot(inputs, weights)

    def forward(self):
        return self.output

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        outputs = [None] * len(self.layers)
        outputs[0] = inputs
        for i in range(1, len(self.layers)):
            neuron = Neuron(outputs[i-1], self.layers[i-1].weights)
            outputs[i] = neuron.forward()
        return outputs[-1]

# Define the layers
layers = [
    Neuron(np.array([1, 2, 3]), np.array([0.5, 0.3, 0.2])),
    Neuron(np.array([4, 5, 6]), np.array([0.4, 0.2, 0.1])),
    Neuron(np.array([7, 8, 9]), np.array([0.3, 0.1, 0.0]))
]

# Create the neural network
network = NeuralNetwork(layers)

# Run the network
inputs = np.array([1, 2, 3])
output = network.forward(inputs)
print(output)