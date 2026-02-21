import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the DNA sequence data
dna_seq_data = pd.read_csv('dna_sequence_data.csv')

# Preprocess the data
scaler = MinMaxScaler()
dna_seq_data[['A', 'C', 'G', 'T']] = scaler.fit_transform(dna_seq_data[['A', 'C', 'G', 'T']])

# Create a neural network model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(dna_seq_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(dna_seq_data[['A', 'C', 'G', 'T']], dna_seq_data['label'], epochs=100, batch_size=32, verbose=0)

# Evaluate the model
loss = model.evaluate(dna_seq_data[['A', 'C', 'G', 'T']], dna_seq_data['label'])
print(f'Test loss: {loss:.4f}')

# Use the trained model to make predictions on new DNA sequences
new_dna_seq_data = pd.read_csv('new_dna_sequence_data.csv')
new_dna_seq_data[['A', 'C', 'G', 'T']] = scaler.transform(new_dna_seq_data[['A', 'C', 'G', 'T']])
predictions = model.predict(new_dna_seq_data[['A', 'C', 'G', 'T']])