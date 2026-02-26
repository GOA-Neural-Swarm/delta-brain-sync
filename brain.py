import numpy as np
class NeuralLayer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    
    def process(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
    
    def optimize(self):
        self.weights -= 0.01 * np.mean(np.dot(np.random.randn(*self.weights.shape), self.weights), axis=0)
        self.biases -= 0.01 * np.mean(np.random.randn(*self.biases.shape), axis=0)