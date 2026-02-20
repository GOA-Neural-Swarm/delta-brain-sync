import numpy as np
import random

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights1 = np.random.rand(inputs, 100)
        self.weights2 = np.random.rand(100, outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs_list, outputs_list):
        inputs = np.array(inputs_list, dtype=float)
        outputs = np.array(outputs_list, dtype=float)
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)
        cost = np.mean((output_layer - outputs) ** 2)
        self.weights1 += inputs.T.dot(hidden_layer) * cost
        self.weights2 += hidden_layer.T.dot(output_layer) * cost

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)
        return output_layer

# Create a neural network with 3 inputs and 1 output
nn = NeuralNetwork(3, 1)

# Train the network
nn.train([[0, 0, 1], [1, 1, 1], [1, 0, 1]], [0, 1, 1])

# Make predictions
print(nn.predict([1, 1, 1]))