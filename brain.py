import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights1 = np.random.rand(hidden, inputs)
        self.weights2 = np.random.rand(outputs, hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            layer1 = self.sigmoid(np.dot(inputs, self.weights1))
            layer2 = self.sigmoid(np.dot(layer1, self.weights2))

            layer2_error = outputs - layer2
            layer2_delta = layer2_error * self.sigmoid_derivative(layer2)

            layer1_error = layer2_delta
            layer1_delta = layer1_error * self.sigmoid_derivative(layer1)

            self.weights2 += layer1_delta.dot(layer1.T)
            self.weights1 += layer1_delta.dot(layer1.T)

    def predict(self, inputs):
        layer1 = self.sigmoid(np.dot(inputs, self.weights1))
        layer2 = self.sigmoid(np.dot(layer1, self.weights2))
        return layer2

# Initialize Neural Network
nn = NeuralNetwork(inputs=2, hidden=5, outputs=1)

# Load Data
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([0, 1, 1, 0])

# Train Neural Network
nn.train(data, target, iterations=5000)

# Test Neural Network
print(nn.predict(np.array([[0, 0]])))
print(nn.predict(np.array([[0, 1]])))
print(nn.predict(np.array([[1, 0]])))
print(nn.predict(np.array([[1, 1]])))

# Plot Neural Network
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()