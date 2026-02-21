import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load DNA sequence data
dna_data = pd.read_csv('dna_sequence_data.csv')

# Preprocess DNA sequence data
scaler = StandardScaler()
dna_data['sequence'] = scaler.fit_transform(dna_data['sequence'])

# Split data into training and testing sets
train_data, test_data = dna_data.split(test_size=0.2, random_state=42)

# Create neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(dna_data.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_data, epochs=10, batch_size=32, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')

# Use model to make predictions on new data
new_data = pd.DataFrame({'sequence': ['ATCGATCG', 'TAGCGTAG', 'CGATCGAT']})
new_data['sequence'] = scaler.transform(new_data['sequence'])
predictions = model.predict(new_data)

# Print predictions
print(predictions)