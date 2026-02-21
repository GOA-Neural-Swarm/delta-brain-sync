import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Define the model architecture
model = Sequential()
model.add(Dense(64, input_shape=(1,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the DNA sequence data
dna_data = pd.read_csv('dna_sequence_data.csv')

# Convert the DNA sequence data into a numerical representation
dna_data['sequence'] = dna_data['sequence'].apply(lambda x: np.array(list(x)))

# Normalize the DNA sequence data
dna_data[['sequence']] = dna_data[['sequence']].apply(lambda x: x / np.max(x))

# Split the data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Train the model on the training data
model.fit(train_data['sequence'], train_data['label'], epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_data['sequence'], test_data['label'])
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Use the trained model to make predictions on new DNA sequences
new_sequence = 'ATCGATCGATCG'
new_sequence = np.array(list(new_sequence)) / np.max(new_sequence)
prediction = model.predict(new_sequence)
print(f'Prediction: {prediction}')