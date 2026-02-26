import numpy as np

class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, neuron_id):
        self.neurons[neuron_id] = {
            'weight': np.random.rand(),
            'input': 0,
            'output': 0
        }

    def activate_neuron(self, neuron_id, input_value):
        self.neurons[neuron_id]['input'] = input_value
        self.neurons[neuron_id]['output'] = self.neurons[neuron_id]['input'] * self.neurons[neuron_id]['weight']

    def get_output(self, neuron_id):
        return self.neurons[neuron_id]['output']

    def reset(self):
        for neuron in self.neurons:
            self.neurons[neuron]['input'] = 0
            self.neurons[neuron]['output'] = 0