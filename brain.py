import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100)  # Initialize weights with random values
        self.biases = np.zeros((100, 1))  # Initialize biases with zeros

    def think(self, input_data):
        # Forward pass
        hidden_layer = np.dot(input_data, self.weights) + self.biases
        hidden_layer = self._sigmoid(hidden_layer)

        output_layer = np.dot(hidden_layer, self.weights.T) + self.biases.T
        output_layer = self._sigmoid(output_layer)

        return output_layer

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def learn(self, input_data, target_output):
        # Backward pass
        hidden_layer = self.think(input_data)
        output_layer = self._sigmoid(np.dot(hidden_layer, self.weights.T) + self.biases.T)

        # Compute error
        error = target_output - output_layer

        # Compute gradients
        d_output_layer = error * output_layer * (1 - output_layer)
        d_hidden_layer = d_output_layer * (hidden_layer * (1 - hidden_layer))

        # Update weights and biases
        self.weights += np.dot(input_data.T, d_hidden_layer) * 0.01
        self.biases += np.sum(d_hidden_layer, axis=0, keepdims=True) * 0.01

brain = Brain()