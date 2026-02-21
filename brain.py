import numpy as np
import tensorflow as tf

class TelefoxX:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

telefox = TelefoxX()