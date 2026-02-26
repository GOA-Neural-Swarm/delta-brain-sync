import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        hidden_layer_output = np.dot(inputs, self.weights1)
        hidden_layer_output = self.sigmoid(hidden_layer_output)
        output_layer_output = np.dot(hidden_layer_output, self.weights2)
        output_layer_output = self.sigmoid(output_layer_output)
        return output_layer_output