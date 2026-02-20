import numpy as np
import matplotlib.pyplot as plt

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = np.random.rand(len(sequence), len(sequence))
        self.biases = np.random.rand(len(sequence), 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            for i in range(len(self.sequence)):
                for j in range(len(self.sequence)):
                    if i!= j:
                        self.weights[i][j] += learning_rate * (self.sigmoid(self.biases[i][0] + np.dot(self.weights[:, j], self.sequence)) - self.sigmoid(self.biases[j][0] + np.dot(self.weights[:, i], self.sequence)))
                        self.biases[i][0] += learning_rate * (self.sigmoid(self.biases[i][0] + np.dot(self.weights[:, j], self.sequence)) - self.sigmoid(self.biases[j][0] + np.dot(self.weights[:, i], self.sequence)))
            self.weights = self.weights / np.linalg.norm(self.weights, axis=0, keepdims=True)
            self.biases = self.biases / np.linalg.norm(self.biases, axis=0, keepdims=True)

    def predict(self, input_sequence):
        output = np.zeros(len(input_sequence))
        for i in range(len(input_sequence)):
            for j in range(len(input_sequence)):
                if i!= j:
                    output[i] += self.weights[i][j] * self.sigmoid(self.biases[i][0] + np.dot(self.weights[:, j], input_sequence))
        return output

brain = Brain(PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)
brain.train(epochs=5000, learning_rate=0.1)
print(brain.predict(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])))