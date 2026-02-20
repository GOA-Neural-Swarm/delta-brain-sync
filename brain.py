import re
import math

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights = [[random.random() for _ in range(hidden)] for _ in range(inputs)]
        self.biases = [random.random() for _ in range(hidden)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def forward_propagate(self, inputs):
        hidden_layer = [self.sigmoid(sum([i * w for i, w in zip(inputs, weights)]) + biases) for weights, biases in zip(self.weights, self.biases)]
        output_layer = [self.sigmoid(sum([h * o for h, o in zip(hidden_layer, self.weights[0])]))]
        return hidden_layer, output_layer

    def backpropagate(self, inputs, target):
        hidden_layer, output_layer = self.forward_propagate(inputs)
        error = target - output_layer[0]
        delta = error * self.derivative_sigmoid(output_layer[0])
        self.weights[0][0] += delta * hidden_layer[0]
        self.biases[0] += delta
        for i in range(1, len(hidden_layer)):
            error = hidden_layer[i-1] * (1 - hidden_layer[i-1]) * delta
            delta *= self.derivative_sigmoid(hidden_layer[i])
            self.weights[i][i-1] += error * inputs[i]
            self.biases[i] += error

    def train(self, inputs, target):
        self.backpropagate(inputs, target)

# Prompt definition closing and Meta-Cognition logic