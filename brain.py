import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.bias = np.zeros(10)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, x):
        z = np.dot(x, self.weights) + self.bias
        a = self.sigmoid(z)
        return a

    def backward_propagation(self, x, a):
        z = np.dot(x, self.weights) + self.bias
        a = self.sigmoid(z)
        d = a * (1 - a)
        dw = x.T.dot(d)
        db = d.sum()
        return dw, db

    def train(self, x, y, epochs=1000):
        for _ in range(epochs):
            a = self.forward_propagation(x)
            dw, db = self.backward_propagation(x, a)
            self.weights -= 0.01 * dw
            self.bias -= 0.01 * db

    def predict(self, x):
        a = self.forward_propagation(x)
        return np.round(a)

nn = NeuralNetwork()
nn.train(np.random.rand(10, 10), np.random.rand(10, 1))
print(nn.predict(np.random.rand(10, 1)))