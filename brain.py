import numpy as np
import tensorflow as tf

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.weights = tf.Variable(tf.random.normal([len(sequence), len(sequence)]))
        self.biases = tf.Variable(tf.random.normal([len(sequence)]))
        self.optimize()

    def optimize(self):
        with tf.GradientTape() as tape:
            output = tf.matmul(self.weights, self.sequence)
            loss = tf.reduce_mean((output - self.sequence) ** 2)
            gradients = tape.gradient(loss, [self.weights, self.biases])
            self.weights.assign_sub(gradients[0])
            self.biases.assign_sub(gradients[1])

    def evolve(self):
        self.optimize()
        new_sequence = tf.matmul(self.weights, self.sequence) + self.biases
        self.sequence = new_sequence
        self.optimize()

brain = Brain(np.array(Source))
for _ in range(1000):
    brain.evolve()
print(brain.sequence)