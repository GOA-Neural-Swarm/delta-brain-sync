import numpy as np
from scipy.optimize import minimize

class Brain:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.synaptic_weights = np.random.rand(num_neurons, num_neurons)

    def think(self, input_array):
        return np.dot(input_array, self.synaptic_weights)

    def learn(self, input_array, output_array):
        def objective(weights):
            return np.mean((np.dot(input_array, weights) - output_array) ** 2)
        result = minimize(objective, self.synaptic_weights.flatten(), method="SLSQP")
        self.synaptic_weights = result.x.reshape(self.synaptic_weights.shape)