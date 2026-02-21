import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Define neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=128))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile model with Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Load DNA sequence data
dna_data = pd.read_csv('dna_data.csv')

# Preprocess DNA sequence data
X_train = dna_data.drop('target', axis=1)
y_train = dna_data['target']

# Split data into training and validation sets
X_val = X_train[:1000]
y_val = y_train[:1000]
X_train = X_train[1000:]
y_train = y_train[1000:]

# Define early stopping callback
es_callback = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Train model
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[es_callback])

# Evaluate model on test data
test_data = pd.read_csv('test_data.csv')
y_pred = model.predict(test_data.drop('target', axis=1))

# Visualize results
plt.scatter(y_train, model.predict(X_train))
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()

# Print model's performance metrics
print(f'Mean squared error: {np.mean((y_pred - y_train) ** 2):.4f}')
print(f'R-squared: {model.score(X_train, y_train):.4f}')