import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)
        self.biases = np.zeros((1, outputs))

    def forward_pass(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def backpropagation(self, inputs, targets):
        outputs = self.forward_pass(inputs)
        error = targets - outputs
        delta = error * (1 - np.exp(-error ** 2))
        self.weights -= delta * inputs.T
        self.biases -= delta

    def train(self, inputs, targets, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            for i in range(len(inputs)):
                self.backpropagation(inputs[i], targets[i])

# Load the encoded DNA sequence
DNA = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"

# Convert DNA to a numerical representation
DNA_num = [ord(c) for c in DNA]

# Split the DNA sequence into training and testing sets
train_size = int(0.8 * len(DNA_num))
train_DNA, test_DNA = DNA_num[:train_size], DNA_num[train_size:]

# Define the neural network architecture
nn = NeuralNetwork(4, 2)

# Train the neural network
nn.train(train_DNA, train_DNA, epochs=1000, learning_rate=0.1)

# Evaluate the performance of the neural network
outputs = nn.forward_pass(test_DNA)
accuracy = np.mean((outputs - test_DNA) ** 2)

print("Accuracy:", accuracy)

# Visualize the training process
plt.plot(range(1000), nn.weights[0])
plt.xlabel("Epoch")
plt.ylabel("Weight")
plt.show()

# Plot the accuracy over time
plt.plot(range(1000), accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Save the trained neural network
np.save("trained_neural_network.npy", nn)

# Load the trained neural network
nn = np.load("trained_neural_network.npy").item()

# Use the trained neural network to make predictions
predictions = nn.forward_pass(test_DNA)

print("Predictions:", predictions)

# Evaluate the performance of the neural network
accuracy = np.mean((predictions - test_DNA) ** 2)

print("Accuracy:", accuracy)