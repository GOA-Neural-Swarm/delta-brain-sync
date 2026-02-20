import random

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [[random.uniform(-1, 1) for _ in range(outputs)] for _ in range(inputs)]

    def calculate_outputs(self, inputs):
        outputs = [0 for _ in range(self.outputs)]
        for i in range(self.inputs):
            for o in range(self.outputs):
                outputs[o] += inputs[i] * self.weights[i][o]
        return outputs

    def mutate(self):
        for i in range(self.inputs):
            for o in range(self.outputs):
                self.weights[i][o] += random.uniform(-0.1, 0.1)

    def evaluate(self, inputs, expected_outputs):
        outputs = self.calculate_outputs(inputs)
        return sum((o - e) ** 2 for o, e in zip(outputs, expected_outputs))

    def evolve(self, inputs, expected_outputs, generations):
        for _ in range(generations):
            self.mutate()
            fitness = self.evaluate(inputs, expected_outputs)
            if fitness < 0.1:
                break
        return self

# Define the neural network
nn = NeuralNetwork(2, 1)

# Train the neural network
inputs = [0, 1]
expected_outputs = [0]
nn.evolve(inputs, expected_outputs, 1000)

# Test the neural network
print(nn.calculate_outputs([0, 1]))  # Should output around 0
print(nn.calculate_outputs([1, 0]))  # Should output around 1