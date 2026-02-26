import numpy as np

class Brain:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        self.firing_neurons = []

    def fire(self, inputs):
        for neuron in self.neurons:
            if neuron.fire(inputs):
                self.firing_neurons.append(neuron)
        return self.firing_neurons

    def propagate(self, firing_neurons):
        for neuron in self.synapses:
            neuron.update(firing_neurons)

    def update_neurons(self, firing_neurons):
        for neuron in firing_neurons:
            neuron.update()

    def reset(self):
        self.firing_neurons = []