import random

class NeuralNetwork:
    def __init__(self):
        self.weights = [random.uniform(-1, 1) for _ in range(100)]
        self.biases = [random.uniform(-1, 1) for _ in range(100)]
        self.input_values = [random.uniform(0, 1) for _ in range(100)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forward_pass(self):
        self.output_values = []
        for i in range(len(self.input_values)):
            weighted_sum = sum([self.weights[i] * self.input_values[i] for i in range(len(self.weights))]) + self.biases[i]
            self.output_values.append(self.sigmoid(weighted_sum))

    def train(self, target_output):
        error = 0
        for i in range(len(self.output_values)):
            error += (self.output_values[i] - target_output[i]) ** 2
        self.error = error / len(self.output_values)
        self.backward_pass()

    def backward_pass(self):
        for i in range(len(self.output_values)):
            delta = (self.output_values[i] - self.target_output[i]) * self.output_values[i] * (1 - self.output_values[i])
            self.weights[i] += delta * self.input_values[i]
            self.biases[i] += delta

    def run(self, target_output):
        self.forward_pass()
        self.train(target_output)

# Neural network initialization
nn = NeuralNetwork()

# Target output for training
target_output = [random.uniform(0, 1) for _ in range(100)]

# Run the neural network
nn.run(target_output)