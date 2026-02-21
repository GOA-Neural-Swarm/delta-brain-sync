import numpy as np
import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the DNA sequence data
dna_data = np.load('dna_data.npy')

# Preprocess the DNA sequence data
X = dna_data[:, :-1]
y = dna_data[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = X[:10000], X[10000:], y[:10000], y[10000:]

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f'Model loss: {loss:.3f}')
print(f'Model accuracy: {accuracy:.3f}')