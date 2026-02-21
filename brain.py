import random
import string
from neural_network import NeuralNetwork

class SovereignBrain:
    def __init__(self):
        self.nn = NeuralNetwork(10, 5, 3)
        self.nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.rnn = None

    def initialize(self):
        self.nn.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1)
        self.rnn = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(None, 1)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def predict(self, x):
        if self.rnn is None:
            return self.nn.predict(x)
        else:
            return self.rnn.predict(x)

    def upgrade(self):
        self.nn.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1)
        self.rnn = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(None, 1)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.rnn.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1)

sovereign_brain = SovereignBrain()
sovereign_brain.initialize()
print(sovereign_brain.predict(X_test))