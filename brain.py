class Brain:
    def __init__(self):
        self.weights = {}
        self.biases = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        learning_rate = 0.1
        for epoch in range(1000):
            for i in range(len(inputs)):
                input_layer = inputs[i]
                target = targets[i]
                predictions = self.forward_pass(input_layer)
                error = target - predictions
                self.backward_pass(input_layer, error, learning_rate)
        return self

    def forward_pass(self, input_layer):
        layer_outputs = {}
        layer_outputs[0] = input_layer
        for i in range(1, len(self.weights) + 1):
            layer_outputs[i] = np.dot(layer_outputs[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            layer_outputs[i] = self.sigmoid(layer_outputs[i])
        return layer_outputs[len(self.weights)]

    def backward_pass(self, input_layer, error, learning_rate):
        layer_outputs = {}
        layer_outputs[0] = input_layer
        for i in range(len(self.weights), 0, -1):
            error = error * self.sigmoid_derivative(layer_outputs[i])
            self.weights[i - 1] += learning_rate * np.dot(layer_outputs[i - 1].T, error)
            self.biases[i - 1] += learning_rate * error
            error = np.dot(error, self.weights[i - 1].T)
        return self