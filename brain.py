import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(1000, 1000)
        self.biases = np.zeros((1000,))
        self.synaptic_weights = np.random.rand(1000, 1000)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        for i in range(len(inputs)):
            layer_1 = np.dot(inputs[i], self.weights) + self.biases
            layer_1 = self.sigmoid(layer_1)
            layer_2 = np.dot(layer_1, self.synaptic_weights)
            layer_2 = self.sigmoid(layer_2)
            layer_2_error = targets[i] - layer_2
            layer_2_delta = layer_2_error * self.sigmoid_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(self.synaptic_weights.T)
            layer_1_delta = layer_1_error * self.sigmoid_derivative(layer_1)
            self.weights += inputs[i].T.dot(layer_1_delta)
            self.biases += layer_1_delta
            self.synaptic_weights += layer_1.T.dot(layer_2_delta)

    def think(self, inputs):
        layer_1 = np.dot(inputs, self.weights) + self.biases
        layer_1 = self.sigmoid(layer_1)
        layer_2 = np.dot(layer_1, self.synaptic_weights)
        layer_2 = self.sigmoid(layer_2)
        return layer_2

# Initialize the neural network
brain = NeuralNetwork()

# Train the network
brain.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))

# Think
print(brain.think(np.array([1, 1])))