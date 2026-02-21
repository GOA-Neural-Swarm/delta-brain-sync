import numpy as np
import matplotlib.pyplot as plt

# Define the DNA sequence data
DNA_data = np.random.rand(100, 1000)  # 100 samples, 1000 features

# Define the neural network architecture
n_inputs = 1000
n_hidden = 50
n_outputs = 100

# Initialize the weights and biases for the hidden layer
weights_hidden = np.random.rand(n_inputs, n_hidden)
biases_hidden = np.zeros((n_hidden,))

# Initialize the weights and biases for the output layer
weights_output = np.random.rand(n_hidden, n_outputs)
biases_output = np.zeros((n_outputs,))

# Define the activation functions for the hidden and output layers
activation_hidden ='relu'
activation_output ='sigmoid'

# Define the loss function and the optimization algorithm
loss_function ='mean_squared_error'
optimizer = 'adam'

# Compile the model
model = Sequential()
model.add(Dense(n_hidden, input_dim=n_inputs, activation=activation_hidden, weights=weights_hidden, biases=biases_hidden))
model.add(Dense(n_outputs, activation=activation_output, weights=weights_output, biases=biases_output))
model.compile(loss=loss_function, optimizer=optimizer)

# Train the model
history = model.fit(DNA_data, epochs=100, batch_size=128)

# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()