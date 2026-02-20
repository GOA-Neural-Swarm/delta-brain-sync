import numpy as np
import tensorflow as tf

class Brain:
    def __init__(self, inputs=128, hidden_units=256, outputs=64):
        self.inputs = inputs
        self.hidden_units = hidden_units
        self.outputs = outputs
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.inputs,)),
            tf.keras.layers.Dense(self.hidden_units, activation='relu'),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        self.model.fit(X, y, epochs=10, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

# Example usage:
brain = Brain()
X = np.random.rand(100, 128)
y = np.random.rand(100, 64)
brain.train(X, y)
print(brain.predict(X))