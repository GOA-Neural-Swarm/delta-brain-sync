import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define the number of neurons in each layer
n_inputs = 784
n_hidden = 256
n_outputs = 10

# Define the number of iterations for training
n_iterations = 1000

# Define the learning rate
lr = 0.01

# Define the activation functions for each layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

# Define the neural network architecture
def neural_network(X, y):
    # Initialize the weights and biases
    weights1 = np.random.rand(n_inputs, n_hidden)
    biases1 = np.zeros((n_hidden,))
    weights2 = np.random.rand(n_hidden, n_outputs)
    biases2 = np.zeros((n_outputs,))

    # Forward pass
    hidden_layer = sigmoid(np.dot(X, weights1) + biases1)
    output_layer = sigmoid(np.dot(hidden_layer, weights2) + biases2)

    # Calculate the loss
    loss = np.mean((output_layer - y) ** 2)

    # Backward pass
    d_output_layer = 2 * (output_layer - y)
    d_hidden_layer = np.dot(d_output_layer, weights2.T) * (hidden_layer * (1 - hidden_layer))

    # Update the weights and biases
    weights1 -= lr * np.dot(X.T, d_hidden_layer) + biases1
    biases1 -= lr * np.sum(d_hidden_layer, axis=0, keepdims=True)
    weights2 -= lr * np.dot(hidden_layer.T, d_output_layer) + biases2
    biases2 -= lr * np.sum(d_output_layer, axis=0, keepdims=True)

    return loss, hidden_layer, output_layer

# Load the training data
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

# Train the neural network
losses = []
for i in range(n_iterations):
    loss, hidden_layer, output_layer = neural_network(train_data, train_labels)
    losses.append(loss)

# Plot the training loss over time
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Load the test data
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# Make predictions on the test data
predictions = np.round(output_layer)

# Evaluate the performance of the neural network
accuracy = accuracy_score(test_labels, predictions)
print(f'Test accuracy: {accuracy:.2f}')