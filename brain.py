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

# Define the sequence data
sequence = np.array([int(i) for i in Source.split(",")])

# Convert the sequence to a binary representation
binary_sequence = []
for i in sequence:
    if i == "A":
        binary_sequence.append(1)
    elif i == "C":
        binary_sequence.append(0)
    elif i == "G":
        binary_sequence.append(1)
    elif i == "T":
        binary_sequence.append(0)

# Convert the binary sequence to a numerical representation
numerical_sequence = np.array(binary_sequence, dtype=np.float32)

# Normalize the sequence data
numerical_sequence = numerical_sequence / np.max(numerical_sequence)

# Train the model
model.fit(numerical_sequence, epochs=100)

# Evaluate the model
loss, accuracy = model.evaluate(numerical_sequence, steps=100)
print(f"Loss: {loss}, Accuracy: {accuracy}")