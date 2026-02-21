import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network model
model = tf.keras.models.Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.random.rand(100, 32), epochs=10, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(np.random.rand(100, 32), verbose=0)
print(f'Test accuracy: {accuracy:.2f}')

# Use the model to predict the output
output = model.predict(np.random.rand(1, 32))
print(f'Predicted output: {output[0]}')