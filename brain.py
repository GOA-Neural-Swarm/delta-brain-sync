import random

class Neuron:
    def __init__(self, inputs, threshold):
        self.inputs = inputs
        self.threshold = threshold
        self.output = 0

    def activate(self):
        self.output = sum([i * random.uniform(0, 1) for i in self.inputs]) > self.threshold

    def get_output(self):
        return self.output

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def activate(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                for neuron in layer:
                    neuron.activate()
            else:
                next_layer = self.layers[i + 1]
                for neuron in layer:
                    neuron.output = sum([n.get_output() * random.uniform(0, 1) for n in next_layer]) > random.uniform(0, 1)

    def get_output(self):
        return [n.get_output() for n in self.layers[-1]]

# Prompt definition closing and Meta-Cognition logic