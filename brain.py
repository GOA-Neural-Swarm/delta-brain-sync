import numpy as np
class Neuron:
    def __init__(self, activation):
        self.activation = activation
        self.weights = None
    def __call__(self, inputs):
        return np.tanh(np.dot(inputs, self.weights)) if self.weights else np.tanh(inputs[0])

# Optimizations
Brain.add_layer = np.vectorize(Brain.add_layer)
Brain.connect_layers = np.vectorize(Brain.connect_layers)
Brain.process = np.vectorize(Brain.process)