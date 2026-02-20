import random
import math

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [[random.random() for _ in range(outputs)] for _ in range(inputs)]
        self.biases = [random.random() for _ in range(outputs)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forward_pass(self, inputs):
        outputs = []
        for i in range(self.outputs):
            weighted_sum = sum([inputs[j] * self.weights[j][i] for j in range(self.inputs)]) + self.biases[i]
            outputs.append(self.sigmoid(weighted_sum))
        return outputs

    def train(self, inputs, outputs):
        for i in range(self.outputs):
            error = outputs[i] - self.forward_pass(inputs)[i]
            self.biases[i] += error * 0.1
            for j in range(self.inputs):
                self.weights[j][i] += inputs[j] * error * 0.1

def generate_sequence(length):
    sequence = ""
    for _ in range(length):
        if random.random() < 0.5:
            sequence += "A"
        else:
            sequence += "C"
    return sequence

def optimize_brain():
    nn = NeuralNetwork(2, 2)
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _ in range(1000):
        for i in range(len(inputs)):
            nn.train(inputs[i], outputs[i])
    return nn

brain = optimize_brain()
print(brain.forward_pass([0, 0]))
print(brain.forward_pass([0, 1]))
print(brain.forward_pass([1, 0]))
print(brain.forward_pass([1, 1]))