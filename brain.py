import numpy as np
import tensorflow as tf

class Brain:
    def __init__(self):
        self.weights = np.random.rand(100, 100)
        self.biases = np.zeros((100,))
        self.neural_network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(100, activation='sigmoid')
        ])

    def think(self, input_data):
        output = np.dot(input_data, self.weights) + self.biases
        return self.neural_network.predict(output)

    def learn(self, input_data, output_data):
        self.weights += np.dot(input_data.T, output_data)
        self.biases += np.mean(output_data, axis=0)

brain = Brain()