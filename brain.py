import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(6, 1)
        self.weights2 = np.random.rand(6, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        learning_rate = 0.01
        iterations = 1000

        for _ in range(iterations):
            z = np.dot(X, self.weights1)
            layer1 = self.sigmoid(z)
            z2 = np.dot(layer1, self.weights2)
            layer2 = self.sigmoid(z2)
            loss = np.mean((layer2 - y) ** 2)
            dloss_dweights2 = np.dot(layer1.T, (2 * (layer2 - y)))
            dloss_dlayer1 = dloss_dweights2 * self.sigmoid_derivative(layer2)
            dloss_dweights1 = np.dot(X.T, dloss_dlayer1 * self.sigmoid_derivative(layer1))
            self.weights1 -= learning_rate * dloss_dweights1
            self.weights2 -= learning_rate * dloss_dweights2

    def predict(self, X):
        X = np.array(X)
        layer1 = self.sigmoid(np.dot(X, self.weights1))
        layer2 = self.sigmoid(np.dot(layer1, self.weights2))
        return layer2

# Initialize the neural network
nn = NeuralNetwork()

# Train the neural network
nn.fit([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0]], [[0], [1], [1], [0], [1], [0]])

# Make predictions
print(nn.predict([[0, 0, 0, 0, 0, 0]]))
print(nn.predict([[0, 0, 1, 1, 1, 1]]))
print(nn.predict([[1, 1, 0, 0, 1, 1]]))
print(nn.predict([[0, 1, 1, 0, 0, 1]]))
print(nn.predict([[0, 0, 0, 1, 1, 1]]))
print(nn.predict([[1, 1, 1, 1, 0, 0]]))