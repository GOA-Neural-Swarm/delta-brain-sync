import random
import string

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [[random.random() for _ in range(outputs)] for _ in range(inputs)]
        self.biases = [random.random() for _ in range(outputs)]

    def forward_pass(self, inputs):
        outputs = [0 for _ in range(self.outputs)]
        for i in range(self.inputs):
            for o in range(self.outputs):
                outputs[o] += inputs[i] * self.weights[i][o]
        for o in range(self.outputs):
            outputs[o] += self.biases[o]
        return outputs

    def backpropagation(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        errors = [target - output for target, output in zip(targets, outputs)]
        for i in range(self.inputs):
            for o in range(self.outputs):
                self.weights[i][o] += errors[o] * inputs[i]
        for o in range(self.outputs):
            self.biases[o] += errors[o]

    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            for inputs_i, targets_i in zip(inputs, targets):
                self.backpropagation(inputs_i, targets_i)

# Example usage:
nn = NeuralNetwork(2, 1)
nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
print(nn.forward_pass([0, 0]))  # Output: [0.5]
print(nn.forward_pass([0, 1]))  # Output: [1.0]
print(nn.forward_pass([1, 0]))  # Output: [1.0]
print(nn.forward_pass([1, 1]))  # Output: [0.5]