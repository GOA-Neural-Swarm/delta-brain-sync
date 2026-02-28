import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Brain:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(784,), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def think(self, inputs):
        inputs = np.array(inputs) / 255.0
        outputs = self.model.predict(inputs)
        return np.argmax(outputs)

brain = Brain()