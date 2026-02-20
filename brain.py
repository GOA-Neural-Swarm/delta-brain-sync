import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, neurons, activation='relu'):
        self.layers.append((neurons, activation))

    def compile(self):
        for i, (neurons, activation) in enumerate(self.layers):
            if i == 0:
                self.layers[i] = (neurons, np.random.rand(neurons, 1))
            else:
                self.layers[i] = (neurons, np.dot(self.layers[i-1][1], np.random.rand(neurons, neurons)))

    def evaluate(self, inputs):
        outputs = inputs
        for i, (neurons, activation) in enumerate(self.layers):
            outputs = np.dot(outputs, self.layers[i][1])
            if activation =='relu':
                outputs = np.maximum(outputs, 0)
            elif activation =='sigmoid':
                outputs = 1 / (1 + np.exp(-outputs))
        return outputs

brain = NeuralNetwork()
brain.add_layer(64, activation='relu')
brain.add_layer(32, activation='relu')
brain.add_layer(8, activation='sigmoid')
brain.compile()

# Prompt definition and Meta-Cognition logic