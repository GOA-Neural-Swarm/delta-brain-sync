import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.weights_i_h = np.random.rand(num_inputs, num_hidden)
        self.weights_h_o = np.random.rand(num_hidden, num_outputs)
        self.biases_i_h = np.zeros((1, num_hidden))
        self.biases_h_o = np.zeros((1, num_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        outputs = self.predict(inputs)
        error = targets - outputs
        self.weights_i_h += np.dot(inputs.T, error * self.sigmoid_derivative(outputs))
        self.weights_h_o += np.dot(error * self.sigmoid_derivative(outputs).reshape(-1, 1), outputs.reshape(1, -1))
        self.biases_i_h += error * self.sigmoid_derivative(outputs)
        self.biases_h_o += error * self.sigmoid_derivative(outputs)

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_i_h) + self.biases_i_h
        hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights_h_o) + self.biases_h_o
        output_layer = self.sigmoid(output_layer)
        return output_layer