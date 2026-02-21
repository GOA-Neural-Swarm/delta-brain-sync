import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load DNA sequence data
dna_data = pd.read_csv('dna_sequence.csv')

# Preprocess data
scaler = StandardScaler()
dna_data[['sequence']] = scaler.fit_transform(dna_data[['sequence']])

# Split data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Create neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(dna_data.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data[['sequence']], train_data['target'], epochs=10, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(test_data[['sequence']], test_data['target'], verbose=0)
print(f'Test accuracy: {accuracy:.2f}')

# Make predictions on new DNA sequence data
new_sequence = pd.DataFrame({'sequence': ['ATCGATCG']})
new_sequence[['sequence']] = scaler.transform(new_sequence[['sequence']])
predictions = model.predict(new_sequence[['sequence']])
print(f'Predicted target value: {predictions:.2f}')