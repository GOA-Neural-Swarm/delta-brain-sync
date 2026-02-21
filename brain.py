import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Load Neon DNA Sequence Analysis Data
neon_data = pd.read_csv('neon_data.csv')

# Preprocess Data
scaler = MinMaxScaler()
neon_data[['Feature1', 'Feature2', 'Feature3']] = scaler.fit_transform(neon_data[['Feature1', 'Feature2', 'Feature3']])

# Split Data into Training and Testing Sets
train_data, test_data = neon_data.split(test_size=0.2, random_state=42)

# Convert Class Labels to Categorical
train_labels = to_categorical(train_data['Class'])
test_labels = to_categorical(test_data['Class'])

# Create Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(neon_data.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Evaluate Model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test Loss: {loss:.2f}')

# Make Predictions
predictions = model.predict(test_data)

# Evaluate Predictions
print(classification_report(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))