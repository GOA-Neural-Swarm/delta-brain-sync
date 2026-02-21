import numpy as np
import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the DNA sequence data
DNA_data = np.load('DNA_sequence.npy')

# Preprocess the data
DNA_data = DNA_data / 255.0

# Train the model
model.fit(DNA_data, epochs=1000)

# Evaluate the model
loss, accuracy = model.evaluate(DNA_data)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Use the model to predict the DNA sequence
predictions = model.predict(DNA_data)

# Print the predictions
print(predictions)