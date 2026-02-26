import numpy as np

class Brain:
    def __init__(self):
        self.weights = np.random.rand()
        self.biases = np.zeros((1,))
        self.neurons = [np.zeros((1,)) for _ in range(1000)]
        self.connections = [(np.random.rand(), np.random.rand()) for _ in range(1000)]

    def think(self, inputs):
        for i in range(len(inputs)):
            self.neurons[i][0] = inputs[i]
        for i in range(len(self.neurons)):
            for j in range(len(self.connections[i][0])):
                self.neurons[i][0] += self.connections[i][0][j] * self.neurons[j][0]
            self.neurons[i][0] += self.biases[i]
        return self.neurons[0][0]

    def learn(self, inputs, outputs):
        for i in range(len(inputs)):
            self.biases[i] += 0.01 * (outputs[i] - self.neurons[i][0])
            for j in range(len(self.connections[i][0])):
                self.connections[i][0][j] += 0.01 * (outputs[i] - self.neurons[i][0]) * self.neurons[j][0]