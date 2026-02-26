class Brain:
    def __init__(self, neurons=1000, synapses=100000):
        self.neurons = neurons
        self.synapses = synapses
        self.activations = [0] * neurons
        self.weights = [[0.0] * (neurons + 1) for _ in range(neurons)]
        self.biases = [0.0] * neurons

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def propagate(self, inputs):
        for i in range(self.neurons):
            self.activations[i] = self.sigmoid(sum(self.weights[i][j] * inputs[j] for j in range(self.neurons)) + self.biases[i])

    def backpropagate(self, targets):
        errors = [0.0] * self.neurons
        for i in range(self.neurons):
            errors[i] = self.activations[i] - targets[i]

    def optimize(self, inputs, targets):
        for i in range(self.neurons):
            for j in range(self.neurons + 1):
                self.weights[i][j] += 0.1 * errors[i] * inputs[j]
                self.biases[i] += 0.1 * errors[i]

    def run(self, inputs):
        self.propagate(inputs)
        return self.activations