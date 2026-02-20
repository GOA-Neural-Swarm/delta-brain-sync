import random

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [[random.random() for _ in range(outputs)] for _ in range(inputs)]
        self.biases = [random.random() for _ in range(outputs)]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def propagate(self, inputs):
        outputs = [self.sigmoid(sum([input * weight for input, weight in zip(inputs, self.weights[i])]) + self.biases[i]) for i in range(self.outputs)]
        return outputs

    def mutate(self):
        for i in range(self.inputs):
            for j in range(self.outputs):
                self.weights[i][j] += random.uniform(-0.1, 0.1)

    def evolve(self, target_outputs):
        inputs = [[random.random() for _ in range(self.inputs)] for _ in range(1000)]
        for _ in range(1000):
            outputs = self.propagate(inputs)
            if all([output >= target_outputs[i] for i, output in enumerate(outputs)]):
                return
            self.mutate()

    def train(self, target_outputs):
        inputs = [[random.random() for _ in range(self.inputs)] for _ in range(1000)]
        for _ in range(1000):
            outputs = self.propagate(inputs)
            if all([output >= target_outputs[i] for i, output in enumerate(outputs)]):
                return
            self.evolve(target_outputs)

# Initialize the neural network with 2 inputs and 2 outputs
brain = NeuralNetwork(2, 2)

# Define the target outputs for the neural network to learn
target_outputs = [0.5, 0.5]

# Train the neural network
brain.train(target_outputs)

# Print the trained weights and biases
print("Weights:")
for i in range(brain.inputs):
    print(brain.weights[i])
print("Biases:")
print(brain.biases)