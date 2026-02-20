import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(self.inputs, self.outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets, learning_rate=0.1, iterations=1000):
        for _ in range(iterations):
            outputs = self.forward_pass(inputs)
            error = np.sum((outputs - targets) ** 2)
            self.backpropagate(inputs, targets, learning_rate)
        return error

    def forward_pass(self, inputs):
        outputs = np.zeros(self.outputs)
        for i in range(self.inputs):
            outputs += self.weights[i] * inputs[i]
        return self.sigmoid(outputs)

    def backpropagate(self, inputs, targets, learning_rate):
        outputs = self.forward_pass(inputs)
        error = np.sum((outputs - targets) ** 2)
        d_error = 2 * (outputs - targets)
        d_outputs = self.sigmoid_derivative(outputs)
        d_error_outputs = d_error * d_outputs
        for i in range(self.inputs):
            self.weights[i] -= learning_rate * d_error_outputs * inputs[i]

# Create the neural network with 5 inputs and 3 outputs
nn = NeuralNetwork(5, 3)

# Train the network
nn.train(np.array([[0, 0, 1, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1]]), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]]))

# Test the network
print(nn.forward_pass(np.array([[0, 0, 1, 1, 0]])))