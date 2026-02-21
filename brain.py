import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Define the RNA sequence
rna_sequence = "ATCGATCGATCG"

# Define the neural network architecture
model = tf.keras.models.Sequential([
    Dense(128, activation='relu', input_shape=(rna_sequence.shape[0], 1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate synthetic data
X_train = np.random.rand(100, rna_sequence.shape[0], 1)
y_train = np.random.randint(0, 2, 100)

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

# Use the trained model to predict on new data
new_data = np.random.rand(1, rna_sequence.shape[0], 1)
prediction = model.predict(new_data)
print(f"Prediction: {prediction:.3f}")