import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load Neon DNA Sequence Data
neon_data = pd.read_csv('neon_data.csv')

# Standardize Features
scaler = StandardScaler()
neon_data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(neon_data[['feature1', 'feature2', 'feature3']])

# Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(neon_data.drop('target', axis=1), neon_data['target'], test_size=0.2, random_state=42)

# Create Neural Network Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))

# Evaluate Model
y_pred = model.predict(X_test)
print('MSE:', np.mean((y_pred - y_test) ** 2))