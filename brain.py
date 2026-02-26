import numpy as np

class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, name, init_value=0):
        self.neurons[name] = {'value': init_value, 'weights': {}}

    def connect(self, neuron1, neuron2, weight):
        if neuron1 in self.neurons and neuron2 in self.neurons:
            self.neurons[neuron1]['weights'][neuron2] = weight
            self.neurons[neuron2]['weights'][neuron1] = weight

    def process(self, neuron_name):
        if neuron_name in self.neurons:
            neuron = self.neurons[neuron_name]
            neuron['value'] = np.sum([neuron['value'] * weight for weight in neuron['weights'].values()])
            return neuron['value']