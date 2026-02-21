import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Define the neural network model
model = tf.keras.models.Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Load the DNA sequence data
dna_data = np.loadtxt('dna_sequence_data.txt')

# Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(dna_data[:, :-1], dna_data[:, -1], test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'MSE: {mse:.2f}')

# Predict the output
output = model.predict(X_test)

# Print the predicted output
print(output)