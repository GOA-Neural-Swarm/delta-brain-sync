import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load DNA sequence data
dna_data = pd.read_csv('neon_dna_sequence.csv')

# Preprocess data
scaler = StandardScaler()
dna_data[['AT', 'CG', 'GC', 'TT']] = scaler.fit_transform(dna_data[['AT', 'CG', 'GC', 'TT']])

# Split data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Build neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(dna_data.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, epochs=100, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')