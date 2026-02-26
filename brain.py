import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.weights_ih = np.random.rand(num_inputs, num_hidden)
        self.weights_ho = np.random.rand(num_hidden, num_outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        for _ in range(1000):  # number of iterations
            hidden_layer = np.dot(inputs, self.weights_ih)
            hidden_layer = self.sigmoid(hidden_layer)

            output_layer = np.dot(hidden_layer, self.weights_ho)
            output_layer = self.sigmoid(output_layer)

            error = outputs - output_layer
            self.weights_ho += np.dot(hidden_layer.T, error) * 0.01
            self.weights_ih += np.dot(inputs.T, hidden_layer.T) * 0.01

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_ih)
        hidden_layer = self.sigmoid(hidden_layer)

        output_layer = np.dot(hidden_layer, self.weights_ho)
        output_layer = self.sigmoid(output_layer)

        return output_layer