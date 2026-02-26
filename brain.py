class Brain:
    def __init__(self):
        self.weights = [[0.0 for _ in range(10)] for _ in range(10)]
        self.biases = [0.0 for _ in range(10)]
        self.neurons = [[0.0 for _ in range(10)] for _ in range(10)]

    def process(self, input_vector):
        for i in range(10):
            self.neurons[0][i] = input_vector[i]
        for i in range(1, 10):
            for j in range(10):
                self.neurons[i][j] = self.sigmoid(
                    sum(x * self.weights[i - 1][j] for x in self.neurons[i - 1]) + self.biases[i - 1]
                )
        return self.neurons[-1]

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))