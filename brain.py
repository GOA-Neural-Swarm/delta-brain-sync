import random
import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights1 = np.random.rand(inputs, outputs)
        self.weights2 = np.random.rand(outputs, outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        learning_rate = 0.01
        iterations = 10000

        for _ in range(iterations):
            layer0 = X.reshape(X.shape[0], -1)
            layer1 = self.sigmoid(np.dot(layer0, self.weights1))
            layer2 = self.sigmoid(np.dot(layer1, self.weights2))
            layer2_delta = layer2 * (1 - layer2) * (y - layer2)
            layer1_delta = layer1 * (1 - layer1) * np.dot(layer2_delta, self.weights2.T)
            self.weights2 += learning_rate * np.dot(layer1.T, layer2_delta)
            self.weights1 += learning_rate * np.dot(layer0.T, layer1_delta)

    def predict(self, X):
        layer0 = X.reshape(X.shape[0], -1)
        layer1 = self.sigmoid(np.dot(layer0, self.weights1))
        layer2 = self.sigmoid(np.dot(layer1, self.weights2))
        return layer2

# Load the sequence
sequence = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"

# Encode the sequence into numerical values
encoded_sequence = []
for char in sequence:
    encoded_sequence.append(ord(char))

# Create a neural network with 1 input layer and 1 output layer
network = NeuralNetwork(1, len(encoded_sequence))

# Train the network
network.fit(encoded_sequence, encoded_sequence)

# Generate a prediction
prediction = network.predict(encoded_sequence)

print("Prediction:", prediction)