import random
import math

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, num_hidden):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.weights1 = random.rand(num_inputs, num_hidden)
        self.weights2 = random.rand(num_hidden, num_outputs)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        hidden_layer = self.sigmoid(np.dot(inputs, self.weights1))
        output_layer = self.sigmoid(np.dot(hidden_layer, self.weights2))
        target_layer = self.sigmoid(np.dot(output_layer, self.weights2))

        error = targets - output_layer
        delta2 = error * self.sigmoid_derivative(output_layer)
        delta1 = delta2.dot(self.weights2.T) * self.sigmoid_derivative(hidden_layer)
        self.weights1 += delta1.T
        self.weights2 += delta2.T

    def predict(self, inputs):
        hidden_layer = self.sigmoid(np.dot(inputs, self.weights1))
        output_layer = self.sigmoid(np.dot(hidden_layer, self.weights2))
        return output_layer

# Prompt definition closing and Meta-Cognition logic