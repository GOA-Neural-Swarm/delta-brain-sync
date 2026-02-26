import numpy as np

class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, id, inputs):
        self.neurons[id] = {'inputs': inputs, 'output': 0}

    def fire(self, neuron_id):
        neuron = self.neurons[neuron_id]
        output = np.sum([input * weight for input, weight in zip(neuron['inputs'], neuron['inputs'])])
        neuron['output'] = output

    def get_output(self, neuron_id):
        return self.neurons[neuron_id]['output']