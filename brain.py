import numpy as np

# Define the neural network structure
nn = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Define the activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the loss function
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Train the neural network
for _ in range(1000):
    # Forward pass
    y_pred = sigmoid(np.dot(nn, np.random.rand()))
    # Backward pass
    delta = mse(y, y_pred) * (y_pred - y)
    # Update the weights
    nn += delta

# Use the trained neural network for prediction
def predict(x):
    return sigmoid(np.dot(nn, x))