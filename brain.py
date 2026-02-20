import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)

        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)

        error = outputs - output_layer
        d_error = error * self.sigmoid_derivative(output_layer)

        self.weights2 += np.dot(hidden_layer.T, d_error)

        delta_weights2 = d_error * hidden_layer
        delta_weights1 = delta_weights2 * self.sigmoid_derivative(hidden_layer)

        self.weights1 += np.dot(inputs.T, delta_weights1)

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        hidden_layer = self.sigmoid(hidden_layer)

        output_layer = np.dot(hidden_layer, self.weights2)
        output_layer = self.sigmoid(output_layer)

        return output_layer

# Initialize Neural Network
nn = NeuralNetwork(4, 2, 1)

# Define inputs and outputs
inputs = np.array([[0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 0]])
outputs = np.array([[0], [1], [1], [0]])

# Train Neural Network
for _ in range(10000):
    nn.train(inputs, outputs)

# Predict outputs
predictions = nn.predict(inputs)

print("Predicted outputs:")
print(predictions)