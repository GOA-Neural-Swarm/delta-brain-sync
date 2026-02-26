import numpy as np

class Brain:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.layers = []

    def add_layer(self, neurons, activation):
        self.layers.append((neurons, activation))
        self.weights.append(np.random.rand(neurons, sum([n for n, _ in self.layers[:-1]])))
        self.biases.append(np.zeros((neurons,)))

    def forward_pass(self, inputs):
        outputs = np.copy(inputs)
        for i, (neurons, activation) in enumerate(self.layers):
            weights = self.weights[i]
            biases = self.biases[i]
            if i == 0:
                outputs = np.dot(weights, outputs) + biases
            else:
                outputs = activation(outputs)
        return outputs

    def backpropagation(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        errors = np.array([(target - output) * output * (1 - output) for target, output in zip(targets, outputs)])
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                errors = np.dot(errors, np.transpose(self.weights[i]))
            else:
                weights = self.weights[i]
                outputs = np.dot(weights, outputs) + self.biases[i]
                errors = np.dot(errors, np.transpose(weights)) * [f(x) for x, f in zip(outputs, [f for _, f in self.layers[i + 1]])]
        return errors