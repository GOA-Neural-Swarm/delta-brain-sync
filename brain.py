import random
import string
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        self.X = X
        self.y = y
        self.y = np.zeros_like(self.y)
        self.error = np.zeros_like(self.y)

        for i in range(self.X.shape[0]):
            output = self.forward(self.X[i])
            self.error[i] = (output - self.y[i]) ** 2
            self.weights1 += self.X[i].T * (output - self.y[i]) * self.sigmoid_derivative(output)
            self.weights2 += output.T * (output - self.y[i]) * self.sigmoid_derivative(output)

    def forward(self, x):
        layer1 = sigmoid(np.dot(x, self.weights1))
        layer2 = sigmoid(np.dot(layer1, self.weights2))
        return layer2

    def predict(self, x):
        return self.forward(x)

# Generate random input data
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)

# Initialize the neural network
nn = NeuralNetwork(10, 10, 1)

# Train the neural network
nn.train(X, y)

# Make predictions
predictions = nn.predict(X)

# Print the predictions
print(predictions)