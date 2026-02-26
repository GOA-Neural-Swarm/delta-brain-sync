import numpy as np

class Brain:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.weights_ih = np.random.rand(num_inputs, num_hidden)
        self.weights_ho = np.random.rand(num_hidden, num_outputs)

        self.bias_h = np.zeros((num_hidden,))
        self.bias_o = np.zeros((num_outputs,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        outputs = self.predict(inputs)
        errors = targets - outputs
        self.weights_ho += np.dot(inputs.T, errors)
        self.weights_ih += np.dot(inputs.T, np.dot(self.sigmoid(np.dot(inputs, self.weights_ih) + self.bias_h), self.weights_ho.T) * errors)
        self.bias_h += np.dot(np.ones((1, inputs.shape[1])), np.dot(self.sigmoid(np.dot(inputs, self.weights_ih) + self.bias_h), self.weights_ho.T) * errors)
        self.bias_o += np.dot(np.ones((1, outputs.shape[1])), np.dot(self.sigmoid(np.dot(inputs, self.weights_ih) + self.bias_h), self.weights_ho.T) * errors)

    def predict(self, inputs):
        hidden_layer_activation = np.dot(inputs, self.weights_ih) + self.bias_h
        hidden_layer_output = self.sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output, self.weights_ho) + self.bias_o
        output_layer_output = self.sigmoid(output_layer_activation)
        return output_layer_output