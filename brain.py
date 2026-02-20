import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights1 = np.random.rand(self.inputs, self.hidden)
        self.weights2 = np.random.rand(self.hidden, self.outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        self.layer2 = self.sigmoid(np.dot(inputs, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer2, self.weights2))

    def backward_pass(self, targets):
        d_output = np.multiply(-2 * (targets - self.output), self.derivative_sigmoid(self.output))
        d_weights2 = np.dot(self.layer2.T, d_output)
        d_layer2 = np.multiply(d_output, self.derivative_sigmoid(self.layer2))
        d_weights1 = np.dot(self.inputs.T, d_layer2)
        return d_weights1, d_weights2

    def train(self, inputs, targets, iterations):
        for _ in range(iterations):
            self.forward_pass(inputs)
            d_weights1, d_weights2 = self.backward_pass(targets)
            self.weights1 += d_weights1
            self.weights2 += d_weights2

nn = NeuralNetwork(2, 3, 1)
nn.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1]]), 1000)