import re
import random
import math

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs * num_hidden)]
        self.bias = [random.uniform(-1, 1) for _ in range(num_hidden)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        hidden_layer = [self.sigmoid(sum(i * w for i, w in zip(inputs, self.weights[j * self.num_inputs:(j + 1) * self.num_inputs]))) + self.bias[j] for j in range(self.num_hidden)]
        return self.sigmoid(sum(i * w for i, w in zip(hidden_layer, self.weights[self.num_hidden * self.num_inputs:])))