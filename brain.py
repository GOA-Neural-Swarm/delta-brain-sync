import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load data
data = pd.read_csv('neon_dna_sequence_analysis.csv')

# Preprocessing
scaler = StandardScaler()
data[['sequence']] = scaler.fit_transform(data[['sequence']])
data.dropna(inplace=True)

# Feature extraction using PCA
pca = PCA(n_components=5)
data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']] = pca.fit_transform(data[['sequence']])

# Model definition
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(5,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
history = model.fit(data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']], data['target'], epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
model.evaluate(data[['pca1', 'pca2', 'pca3', 'pca4', 'pca5']], data['target'])