import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)
        self.biases1 = np.zeros((num_hidden,))
        self.biases2 = np.zeros((num_outputs,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1) + self.biases1
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights2) + self.biases2
        output_layer = self.sigmoid(output_layer)
        return output_layer

    def train(self, inputs, targets, learning_rate, epochs):
        for _ in range(epochs):
            hidden_layer = np.dot(inputs, self.weights1) + self.biases1
            hidden_layer = self.sigmoid(hidden_layer)
            output_layer = np.dot(hidden_layer, self.weights2) + self.biases2
            output_layer = self.sigmoid(output_layer)

            # Calculate error
            error = np.mean((output_layer - targets) ** 2)

            # Calculate gradients
            gradients = 2 * (output_layer - targets)
            gradients_hidden = gradients.dot(self.weights2.T)

            # Update weights and biases
            self.weights2 -= learning_rate * gradients.dot(hidden_layer.T)
            self.weights1 -= learning_rate * gradients_hidden.dot(inputs.T)
            self.biases2 -= learning_rate * np.mean(gradients, axis=0, keepdims=True)
            self.biases1 -= learning_rate * np.mean(gradients_hidden, axis=0, keepdims=True)

    def predict(self, inputs):
        return self.forward_pass(inputs)