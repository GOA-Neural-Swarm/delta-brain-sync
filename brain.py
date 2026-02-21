import numpy as np
import pandas as pd

# Load the neon DNA sequence data
dna_data = pd.read_csv('neon_dna_sequence.csv')

# Preprocess the data by removing rows with missing values
dna_data = dna_data.dropna()

# Split the data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(dna_data.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, batch_size=128, validation_data=test_data)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')