import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Load neon DNA sequence data
neon_data = pd.read_csv("neon_dna.csv")

# Preprocess data
X = neon_data.drop(["target"], axis=1)
y = neon_data["target"]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Make predictions on testing data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy[1]}")