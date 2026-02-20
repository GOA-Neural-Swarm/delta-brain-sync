import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
        self.output = 0

    def compute_output(self, inputs):
        self.output = np.dot(inputs, self.w) + self.x
        return self.output

    def plot(self):
        plt.scatter(self.x, self.output)
        plt.show()

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def compute_output(self, inputs):
        outputs = []
        for layer in self.layers:
            neuron_outputs = []
            for neuron in layer:
                output = neuron.compute_output(inputs)
                neuron_outputs.append(output)
            outputs.append(neuron_outputs)
            inputs = neuron_outputs
        return outputs

    def plot(self):
        for i, layer in enumerate(self.layers):
            plt.subplot(len(self.layers), 1, i + 1)
            for neuron in layer:
                neuron.plot()
        plt.show()

# Generate the sequence
dna = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"
sequence = [ord(c) for c in dna]

# Create the neural network
layers = []
for i in range(len(sequence)):
    if i % 3 == 0:
        layers.append([Neuron(i / 3, i, 0.1)])

# Train the network
network = NeuralNetwork(layers)
network.plot()