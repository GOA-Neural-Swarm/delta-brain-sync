import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load DNA sequence data
dna_data = pd.read_csv('neon_dna_sequence.csv')

# Preprocess DNA sequence data
scaler = StandardScaler()
dna_data[['ATCG']] = scaler.fit_transform(dna_data[['ATCG']])

# Define neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile neural network model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train neural network model
model.fit(dna_data[['ATCG']], dna_data['label'], epochs=10, batch_size=32, verbose=2)

# Make predictions
predictions = model.predict(dna_data[['ATCG']])

# Evaluate model performance
accuracy = model.evaluate(dna_data[['ATCG']], dna_data['label'], verbose=0)
print(f'Model accuracy: {accuracy[1]:.2f}')