import numpy as np
from scipy.optimize import minimize

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons
        self.synaptic_weights = np.random.rand(neurons, neurons)
        self.synaptic_thresholds = np.random.rand(neurons)

    def fire(self, input_array):
        output_array = np.zeros(self.neurons)
        for i in range(self.neurons):
            output_array[i] = self.calculate_output(i, input_array)
        return output_array

    def calculate_output(self, neuron_index, input_array):
        sum_of_inputs = np.dot(self.synaptic_weights[neuron_index], input_array)
        output = np.where(sum_of_inputs > self.synaptic_thresholds[neuron_index], 1, 0)
        return output

    def learn(self, input_array, target_output):
        error = np.mean((self.fire(input_array) - target_output) ** 2)
        gradients = np.zeros((self.neurons, self.neurons))
        for i in range(self.neurons):
            for j in range(self.neurons):
                gradients[i, j] = -2 * (self.fire(input_array)[i] - target_output[i]) * self.fire(input_array)[j]
        self.synaptic_weights -= 0.1 * gradients
        return error

    def optimize(self, input_array, target_output):
        error = np.inf
        while error > 1e-6:
            error = self.learn(input_array, target_output)
        return error

    def generate_synaptic_weights(self):
        return self.synaptic_weights