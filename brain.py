class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def process_neural_signal(self, signal):
        # Optimize neural processing by using NumPy arrays for efficient computations
        import numpy as np

        # Initialize neuron activations
        activations = np.zeros((len(self.neurons),))

        # Propagate neural signals through synapses
        for synapse in self.synapses.values():
            weights, bias = synapse
            activations += np.dot(weights, np.transpose(signal)) + bias

        # Apply sigmoid function to neuron activations
        activations = 1 / (1 + np.exp(-activations))

        return activations

    def update_synapses(self, signal):
        # Update synapses using Hebbian learning
        for neuron in self.neurons.values():
            weights = neuron['weights']
            for synapse in self.synapses.values():
                weights += np.dot(synapse[0], signal)  # Update weights
                # Update bias
                bias = neuron['bias']
                synapse[1] += np.sum(signal)  # Update bias

    def add_synapse(self, neuron1, neuron2, weight, bias):
        # Add a new synapse between two neurons
        self.synapses[(neuron1, neuron2)] = (weight, bias)

    def add_neuron(self, neuron_id, activation_function):
        # Add a new neuron to the brain
        self.neurons[neuron_id] = {'activation_function': activation_function}