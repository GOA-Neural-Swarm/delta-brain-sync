# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the DNA sequence
dna_sequence = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"

# Convert DNA to numeric sequence
numeric_sequence = np.array([ord(char) for char in dna_sequence])

# Scale the sequence using StandardScaler
scaler = StandardScaler()
scaled_sequence = scaler.fit_transform(numeric_sequence.reshape(-1, 1))

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(scaled_sequence, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(scaled_sequence, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Predict the output
output = model.predict(scaled_sequence)
print(output)

# Optimize the model
model.optimizer.lr = 0.01
model.fit(scaled_sequence, epochs=1000, verbose=0)

# Evaluate the optimized model
loss, accuracy = model.evaluate(scaled_sequence, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Output the optimized model
print(model.to_json())