import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights1 = np.random.rand(hidden, inputs)
        self.weights2 = np.random.rand(outputs, hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer = np.dot(self.weights1, inputs)
        hidden_layer_activation = self.sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.weights2, hidden_layer_activation)
        output_layer_activation = self.sigmoid(self.output_layer)
        return output_layer_activation

    def train(self, inputs, targets, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            hidden_layer = np.dot(self.weights1, inputs)
            hidden_layer_activation = self.sigmoid(hidden_layer)
            output_layer = np.dot(self.weights2, hidden_layer_activation)
            output_layer_activation = self.sigmoid(output_layer)

            error = targets - output_layer_activation
            delta2 = error * self.derivative_sigmoid(output_layer_activation)
            delta1 = np.dot(delta2, self.weights2.T) * self.derivative_sigmoid(hidden_layer_activation)

            self.weights2 += learning_rate * np.dot((delta2 * output_layer_activation * (1 - output_layer_activation)), hidden_layer_activation)
            self.weights1 += learning_rate * np.dot(delta1 * (1 - hidden_layer_activation) * hidden_layer_activation, inputs)

    def predict(self, inputs):
        return self.forward(inputs)

# Define the brain structure
brain = NeuralNetwork(2, 5, 1)

# Train the brain
brain.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))

# Predict using the trained brain
print(brain.predict(np.array([[1, 1]])))