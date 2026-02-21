import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load DNA sequence data
dna_data = pd.read_csv('dna_sequence.csv')

# Preprocess DNA sequence data
dna_data = dna_data.dropna()
dna_data['dna_sequence'] = dna_data['dna_sequence'].apply(lambda x: np.fromstring(x, dtype=int))

# Split data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Create LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(train_data.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM model
model.fit(train_data['dna_sequence'], train_data['target'], epochs=50, batch_size=32, validation_data=(test_data['dna_sequence'], test_data['target']))

# Make predictions on test data
predictions = model.predict(test_data['dna_sequence'])

# Evaluate model performance
mse = model.evaluate(test_data['dna_sequence'], test_data['target'])
print(f'MSE: {mse:.4f}')