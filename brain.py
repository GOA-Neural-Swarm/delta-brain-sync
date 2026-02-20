import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(1, 100)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Training Parameters
epochs = 100
batch_size = 32
train_data = np.random.rand(10000, 100)
train_labels = np.random.randint(0, 2, size=(10000, 1))
test_data = np.random.rand(1000, 100)
test_labels = np.random.randint(0, 2, size=(1000, 1))

# Train Model
model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))

# Evaluate Model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)