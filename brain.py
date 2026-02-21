import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the DNA sequence data
dna_data = pd.read_csv('dna_sequence_data.csv')

# Preprocess the data
X = dna_data.drop('label', axis=1).values
y = dna_data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Use the trained model to make predictions on new DNA sequences
new_dna_sequence = np.array([[0.5, 0.2, 0.8,...]])  # Replace with a new DNA sequence
prediction = model.predict(new_dna_sequence)
print(f'Prediction: {prediction:.2f}')