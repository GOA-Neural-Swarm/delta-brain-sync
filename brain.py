import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Neural Network Architecture
class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)

    def forward_pass(self, inputs):
        outputs = np.dot(inputs, self.weights)
        return outputs

    def backward_pass(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        error = np.mean((outputs - targets) ** 2)
        d_weights = np.dot(inputs.T, 2 * (outputs - targets))
        return error, d_weights

    def train(self, inputs, targets):
        error, d_weights = self.backward_pass(inputs, targets)
        self.weights += 0.01 * d_weights
        return error

# Initialize Neural Network
nn = NeuralNetwork(2, 1)

# Define Training Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Train Neural Network
for _ in range(10000):
    error = nn.train(X, Y)

# Plot Training Data and Network Output
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Training Data and Neural Network Output')
plt.show()

print("Neural Network Trained!")