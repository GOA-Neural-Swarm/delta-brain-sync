import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the neural network architecture
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('neon_dna_sequence_analysis.csv')

# Preprocess the data
scaler = StandardScaler()
data[['sequence']] = scaler.fit_transform(data[['sequence']])

# Split the data into training and testing sets
train_data, test_data = data.split(test_size=0.2, random_state=42)

# Create the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(data.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, verbose=0)

print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')