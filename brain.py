import numpy as np

class Brain:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.activations = []

    def add_layer(self, neurons, activation='sigmoid'):
        self.layers.append(neurons)
        self.weights.append(np.random.rand(neurons, self.layers[-2] if self.layers else 0))
        self.biases.append(np.zeros((neurons, 1)))
        self.activations.append(activation)

    def forward_pass(self, inputs):
        inputs = np.array(inputs).reshape((1, -1))
        for i, (layer, weights, biases, activation) in enumerate(zip(self.layers, self.weights, self.biases, self.activations)):
            if activation =='sigmoid':
                inputs = np.where(inputs > 0.5, 1, 0)
            elif activation =='relu':
                inputs = np.where(inputs > 0, inputs, 0)
        return inputs[0]