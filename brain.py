import numpy as np
import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Load the RNA QT45 Predator Logic dataset
data = np.load('rna_qt45_predator_logic_dataset.npy')

# Split the data into training and testing sets
train_data, test_data = data.split(test_size=0.2, random_state=42)

# Train the model on the training data
model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate the model on the testing data
mse = model.evaluate(test_data)
print(f'MSE: {mse:.2f}')