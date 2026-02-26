import math

class Brain:
    def __init__(self):
        self.neurons = {}
        self.connections = {}

    def add_neuron(self, neuron_id):
        self.neurons[neuron_id] = {'weight': 0.0, 'bias': 0.0, 'output': 0.0}

    def add_connection(self, neuron_id1, neuron_id2, weight):
        self.connections[(neuron_id1, neuron_id2)] = weight

    def update_neurons(self):
        for neuron_id in self.neurons:
            output = self.neurons[neuron_id]['bias']
            for connection in self.connections:
                if connection[0] == neuron_id:
                    output += self.connections[connection]*(self.neurons[connection[1]]['output'])
            self.neurons[neuron_id]['output'] = sigmoid(output)

    def process(self):
        self.update_neurons()
        return '\n'.join(f'Neuron {neuron_id}: output={self.neurons[neuron_id]["output"]}' for neuron_id in self.neurons)