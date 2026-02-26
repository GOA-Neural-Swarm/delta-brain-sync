import random
from brain import Brain, NeuralNetwork, Sigmoid, ReLU

# Create brain
brain = Brain()

# Set number of inputs, outputs, and hidden layers
num_inputs = 3
num_outputs = 2
num_hidden_layers = 2

# Create neural network
neural_network = NeuralNetwork(num_inputs, num_outputs, num_hidden_layers)

# Set weights and biases
for layer in range(num_hidden_layers + 1):
    if layer == 0:
        num_neurons = num_inputs
    else:
        num_neurons = num_outputs

    for neuron in range(num_neurons):
        weights = [random.random() for _ in range(num_inputs)]
        biases = [random.random() for _ in range(num_neurons)]

        neural_network.weights[neuron] = weights
        neural_network.biases[neuron] = biases

# Set sigmoid and ReLU functions
sigmoid = Sigmoid()
relu = ReLU()

# Set activation functions
activation_functions = [sigmoid if layer < num_hidden_layers else relu for layer in range(num_hidden_layers + 1)]

# Set neural network processing
def process(self, input_data: np.ndarray) -> np.ndarray:
    output_data = input_data
    for layer in range(num_hidden_layers + 1):
        for neuron in range(num_neurons):
            if layer == 0:
                output_data[neuron] = np.sum([weight * input_data[j] for j, weight in enumerate(self.weights[neuron])]) + self.biases[neuron]
            else:
                output_data[neuron] = activation_functions[layer](np.sum([weight * output_data[j] for j, weight in enumerate(self.weights[neuron])]) + self.biases[neuron])

    return output_data

# Process input data
output_data = brain.process(input_data)

print(output_data)