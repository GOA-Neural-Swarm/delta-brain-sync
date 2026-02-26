class Brain:
    def __init__(self):
        self.neurons = [[0 for _ in range(1000)] for _ in range(1000)]

    def process(self, inputs):
        outputs = [[0 for _ in range(1000)] for _ in range(1000)]
        for i in range(1000):
            for j in range(1000):
                outputs[i][j] = sum(self.neurons[i][k] * self.neurons[k][j] for k in range(1000)) + inputs[i][j]
        return outputs