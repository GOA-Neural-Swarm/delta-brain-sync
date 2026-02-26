import numpy as np

class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.synaptic_weights = np.random.rand(num_synapses, num_neurons)

    def think(self, input_signal):
        output_signal = np.dot(self.synaptic_weights, input_signal)
        return output_signal

    def learn(self, input_signal, output_signal):
        error = output_signal - self.think(input_signal)
        self.synaptic_weights += error * input_signal