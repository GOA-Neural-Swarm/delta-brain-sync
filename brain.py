import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(1, 10)
        self.biases = np.zeros((1, 10))
        self.learning_rate = 0.1

    def forward_pass(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return outputs

    def backward_pass(self, inputs, targets):
        errors = targets - self.forward_pass(inputs)
        delta = errors * self.learning_rate
        self.weights += delta
        self.biases += delta

    def train(self, inputs, targets):
        for _ in range(1000):
            self.backward_pass(inputs, targets)

    def predict(self, inputs):
        outputs = self.forward_pass(inputs)
        return np.argmax(outputs)

# Initialize neural network
nn = NeuralNetwork()

# Train neural network
nn.train(inputs, targets)

# Make prediction
prediction = nn.predict(inputs)

print("Prediction:", prediction)