import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load neon DNA sequence data
neon_data = pd.read_csv('neon_dna_sequence.csv')

# Split data into training and testing sets
train_data, test_data = neon_data.split(test_size=0.2, random_state=42)

# Scale data using StandardScaler
scaler = StandardScaler()
train_data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(train_data[['feature1', 'feature2', 'feature3']])
test_data[['feature1', 'feature2', 'feature3']] = scaler.transform(test_data[['feature1', 'feature2', 'feature3']])

# Create neural network model
model = Sequential()
model.add(Dense(64, input_shape=(3,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model on training data
model.fit(train_data[['feature1', 'feature2', 'feature3']], train_data['target'], epochs=100, batch_size=32, verbose=0)

# Evaluate model on testing data
loss, accuracy = model.evaluate(test_data[['feature1', 'feature2', 'feature3']], test_data['target'], verbose=0)
print(f'Testing accuracy: {accuracy:.2f}')

# Use trained model to predict on new data
new_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7, 8, 9]})
predictions = model.predict(new_data[['feature1', 'feature2', 'feature3']])

# Print predictions
print(predictions)