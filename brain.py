import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# Define model architecture
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(None, 4)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load Neon DNA Sequence data
neon_data = np.loadtxt('neon_data.txt')

# Split data into training and testing sets
train_data, test_data = neon_data[:int(0.8*len(neon_data))], neon_data[int(0.8*len(neon_data)):]

# Normalize data
train_data = train_data / np.max(train_data, axis=0)
test_data = test_data / np.max(test_data, axis=0)

# Train model
model.fit(train_data, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f'Test accuracy: {accuracy:.2f}')