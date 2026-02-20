import numpy as np
import tensorflow as tf

class NeonDNA(tf.keras.Model):
    def __init__(self):
        super(NeonDNA, self).__init__()
        self.W = np.random.rand(1, 1, 1)
        self.B = np.random.rand(1, 1, 1)
        self.P = np.random.rand(1, 1, 1)

    def call(self, x):
        x = tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.B)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return x

model = NeonDNA()