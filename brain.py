import numpy as np

class Brain:
    def __init__(self, num_neurons, num_layers):
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.weights = [np.random.rand(num_neurons, num_neurons) for _ in range(num_layers)]
        self.biases = [np.zeros((num_neurons,)) for _ in range(num_layers)]

    def forward(self, inputs):
        for i in range(self.num_layers):
            inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
            inputs = np.maximum(inputs, 0)
        return inputs

    def backward(self, inputs, outputs):
        for i in range(self.num_layers - 1, -1, -1):
            error = outputs - inputs
            if i > 0:
                error = np.dot(error, self.weights[i].T)
            self.weights[i] += error[:, None] * inputs[None, :]
            self.biases[i] += error
        return self