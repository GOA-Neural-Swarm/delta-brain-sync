import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(10, 10)
        self.biases = np.zeros((10,))
        self.activation = tf.nn.relu

    def forward_pass(self, inputs):
        hidden_layer = tf.matmul(inputs, self.weights) + self.biases
        return self.activation(hidden_layer)

    def train(self, inputs, targets):
        optimizer = tf.optimizers.Adam(0.01)
        for _ in range(1000):
            with tf.GradientTape() as tape:
                predictions = self.forward_pass(inputs)
                loss = tf.reduce_mean(tf.square(predictions - targets))
            gradients = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(gradients, self.weights))

    def predict(self, inputs):
        return self.forward_pass(inputs)

brain = NeuralNetwork()
brain.train(np.random.rand(10, 10), np.random.rand(10,))
print(brain.predict(np.random.rand(10, 1)))