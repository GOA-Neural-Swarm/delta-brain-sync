import numpy as np
import random

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        outputs = self.sigmoid(np.dot(inputs, self.weights))
        targets = targets.reshape((outputs.size, 1))
        self.weights += np.dot(inputs.T, (outputs - targets) * self.sigmoid_derivative(outputs))

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

def generate_code():
    inputs = random.randint(1, 10)
    outputs = random.randint(1, 10)
    nn = NeuralNetwork(inputs, outputs)
    training_data = np.random.rand(inputs, outputs)
    targets = np.random.rand(outputs, 1)
    nn.train(training_data, targets)
    prediction = nn.predict(training_data)
    return prediction

print(generate_code())