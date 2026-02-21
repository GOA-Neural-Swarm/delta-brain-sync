import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

# Load Data
data = pd.read_csv('neon_dna_sequence_analysis.csv')

# Preprocess Data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(data['sequence'])
sequences = tokenizer.texts_to_sequences(data['sequence'])
padded_sequences = pad_sequences(sequences, maxlen=100)

# Split Data
train_sequences = padded_sequences[:int(0.8*len(padded_sequences))]
test_sequences = padded_sequences[int(0.8*len(padded_sequences)):]

# Create Model
model = Sequential()
model.add(LSTM(128, input_shape=(100, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)
model.fit(train_sequences, epochs=10, validation_data=test_sequences, callbacks=[early_stopping])

# Evaluate Model
loss, accuracy = model.evaluate(test_sequences, test_sequences)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Use Model for Prediction
prediction = model.predict(test_sequences)
print(f'Prediction: {prediction}')