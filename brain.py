import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def add_synapse(self, neuron1, neuron2, weight):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        self.synapses[neuron1][neuron2] = weight

    def add_neuron(self, neuron_id, activation_function):
        self.neurons[neuron_id] = activation_function

    def process_neural_signal(self, signal):
        for neuron_id, activation_function in self.neurons.items():
            if neuron_id in self.synapses:
                for connected_neuron, weight in self.synapses[neuron_id].items():
                    signal += weight * self.neurons[connected_neuron](signal)
            else:
                signal = self.neurons[neuron_id](signal)
        return signal

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)