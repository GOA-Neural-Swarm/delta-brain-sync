import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load DNA sequence data
dna_data = pd.read_csv('neon_dna_sequence.csv')

# Preprocess DNA sequence data
X = dna_data['sequence'].values
y = dna_data['label'].values

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype('int32')

# Evaluate performance
accuracy = np.mean(y_pred_class == y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Use the trained model to predict the label for a new DNA sequence
new_sequence = 'ATCGATCGATCG'
new_sequence_array = np.array([new_sequence])

# Predict the label
new_label = model.predict(new_sequence_array)
new_label_class = (new_label > 0.5).astype('int32')

print(f'Predicted label for new sequence: {new_label_class}')