import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)

    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        output_layer = np.dot(hidden_layer, self.weights2)
        return output_layer

    def train(self, inputs, targets):
        hidden_layer = np.dot(inputs, self.weights1)
        output_layer = np.dot(hidden_layer, self.weights2)
        output_errors = targets - output_layer
        hidden_errors = np.dot(output_errors, self.weights2.T)
        self.weights1 += np.dot(inputs.T, hidden_errors)
        self.weights2 += np.dot(hidden_layer.T, output_errors)

    def predict(self, inputs):
        return self.forward(inputs)