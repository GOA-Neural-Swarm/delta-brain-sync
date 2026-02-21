import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(10, 10)

    def forward_pass(self, inputs):
        outputs = np.dot(inputs, self.weights)
        return outputs

    def backpropagation(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        error = targets - outputs
        delta = np.dot(error.T, error)
        self.weights -= delta
        return delta

    def train(self, inputs, targets):
        delta = self.backpropagation(inputs, targets)
        return delta

    def predict(self, inputs):
        outputs = self.forward_pass(inputs)
        return np.argmax(outputs)

nn = NeuralNetwork()
inputs = np.random.rand(10, 10)
targets = np.random.rand(10, 10)
delta = nn.train(inputs, targets)
print(nn.predict(inputs))