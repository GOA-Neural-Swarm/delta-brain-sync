import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define neural network architecture
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Load DNA sequence data
dna_data = pd.read_csv('dna_sequence.csv')

# Preprocess DNA sequence data
scaler = MinMaxScaler()
dna_data[['sequence']] = scaler.fit_transform(dna_data[['sequence']])

# Split data into training and testing sets
train_size = int(0.8 * len(dna_data))
train_data, test_data = dna_data[0:train_size], dna_data[train_size:]

# Create training and testing datasets
X_train = train_data[['sequence']].values
y_train = train_data['target'].values
X_test = test_data[['sequence']].values
y_test = test_data['target'].values

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
print(f'MSE: {mse:.2f}')

# Use the model to make predictions on new DNA sequences
new_sequences = pd.DataFrame({'sequence': ['ATCG', 'CGAT', 'GATC', 'TAGC']})
new_sequences[['sequence']] = scaler.fit_transform(new_sequences[['sequence']])
new_sequences['target'] = model.predict(new_sequences[['sequence']])

# Print the predictions
print(new_sequences)