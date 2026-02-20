import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs, outputs)
        self.bias = np.zeros(outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        predictions = self.predict(inputs)
        error = targets - predictions
        self.weights += np.dot(inputs.T, error)
        self.bias += error

    def predict(self, inputs):
        inputs = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(inputs)

# Define the sequence
sequence = 'MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD'

# Convert the sequence to binary
binary_sequence = ''.join(format(ord(char), '08b') for char in sequence)

# Split the binary sequence into chunks of 8 bits (1 byte)
chunks = [binary_sequence[i:i+8] for i in range(0, len(binary_sequence), 8)]

# Convert each chunk to an integer
int_chunks = [int(chunk, 2) for chunk in chunks]

# Define the inputs and outputs for the neural network
inputs = np.array([[int_chunk] for int_chunk in int_chunks])
outputs = np.array([[0.0]])

# Create and train the neural network
nn = NeuralNetwork(len(chunks), 1)
for _ in range(1000):
    nn.train(inputs, outputs)

# Use the trained neural network to predict the next chunk
next_chunk = nn.predict(inputs)[0]

# Convert the predicted chunk to binary and then to a character
binary_next_chunk = format(int(next_chunk), '08b')
next_char = chr(int(binary_next_chunk, 2))

print(next_char)