import numpy as np
import tensorflow as tf

class Brain:
    def __init__(self):
        self.nn_weights = []
        self.nn_biases = []

    def load_neural_network(self, sequence):
        for i in range(0, len(sequence), 2):
            weight = np.array([int(sequence[i+1], 16)])
            bias = np.array([int(sequence[i], 16)])
            self.nn_weights.append(weight)
            self.nn_biases.append(bias)

    def calculate_output(self, inputs):
        outputs = []
        for i in range(len(self.nn_weights)):
            output = np.dot(inputs, self.nn_weights[i]) + self.nn_biases[i]
            outputs.append(tf.nn.relu(output))
        return outputs

    def train(self, inputs, targets):
        self.load_neural_network(inputs)
        for i in range(len(self.nn_weights)):
            self.nn_weights[i] = tf.Variable(self.nn_weights[i], dtype=tf.float32)
            self.nn_biases[i] = tf.Variable(self.nn_biases[i], dtype=tf.float32)
        optimizer = tf.optimizers.Adam(0.001)
        loss_fn = tf.keras.losses.MeanSquaredError()
        for epoch in range(100):
            with tf.GradientTape() as tape:
                outputs = self.calculate_output(inputs)
                loss = loss_fn(tf.stack(outputs), targets)
                gradients = tape.gradient(loss, self.nn_weights + self.nn_biases)
                optimizer.apply_gradients(zip(gradients, self.nn_weights + self.nn_biases))
        print("Training complete")

    def run(self, inputs):
        outputs = self.calculate_output(inputs)
        return tf.stack(outputs)

# Example usage:
brain = Brain()
inputs = np.array([[1, 2, 3, 4, 5]])
targets = np.array([[10, 20, 30, 40, 50]])
brain.train(inputs, targets)
outputs = brain.run(inputs)
print(outputs)