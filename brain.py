import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Define the sequence
sequence = 'MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD'

# Convert the sequence to a numerical array
sequence_array = np.array([ord(char) for char in sequence])

# Normalize the sequence array using StandardScaler
scaler = StandardScaler()
sequence_array_scaled = scaler.fit_transform(sequence_array.reshape(-1, 1))

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_shape=(1,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(sequence)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(sequence_array_scaled, epochs=100)

# Predict the next character in the sequence
next_char = model.predict(sequence_array_scaled[-1].reshape(1, 1))[0].argmax()

# Print the predicted next character
print(f'Predicted next character: {chr(next_char)}')

# Plot the sequence and predicted next character
plt.plot(sequence_array_scaled)
plt.axvline(len(sequence_array_scaled) - 1, color='r', linestyle='--')
plt.xlabel('Sequence Index')
plt.ylabel('Sequence Value')
plt.title('Sequence and Predicted Next Character')
plt.show()