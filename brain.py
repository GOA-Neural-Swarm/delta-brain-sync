import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load DNA Sequence Data
dna_data = pd.read_csv('dna_sequence_data.csv')

# Preprocess DNA Sequence Data
scaler = StandardScaler()
dna_data['sequence'] = scaler.fit_transform(dna_data['sequence'])

# Split Data into Training and Testing Sets
train_size = int(0.8 * len(dna_data))
train_data, test_data = dna_data[0:train_size], dna_data[train_size:]

# Convert Sequence Data into NumPy Arrays
train_sequences = train_data['sequence'].values
test_sequences = test_data['sequence'].values

# Convert Sequence Data into One-Hot Encoded Format
train_sequences_onehot = to_categorical(train_sequences)
test_sequences_onehot = to_categorical(test_sequences)

# Define LSTM Model
model = Sequential()
model.add(LSTM(64, input_shape=(train_sequences.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(train_sequences_onehot, epochs=10, batch_size=32, validation_data=test_sequences_onehot)

# Evaluate Model
loss, accuracy = model.evaluate(test_sequences_onehot)
print(f'Test loss: {loss:.2f}, Test accuracy: {accuracy:.2f}')