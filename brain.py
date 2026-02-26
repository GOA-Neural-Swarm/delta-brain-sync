import numpy as np

class Brain:
    def __init__(self):
        self.neurons = []
        self.connections = {}

    def add_neuron(self, neuron_id):
        if neuron_id not in self.neurons:
            self.neurons.append(neuron_id)
            self.connections[neuron_id] = {}

    def connect_neurons(self, neuron_id1, neuron_id2, weight):
        if neuron_id1 in self.neurons and neuron_id2 in self.neurons:
            self.connections[neuron_id1][neuron_id2] = weight

    def process(self, inputs):
        for neuron_id, input_value in inputs.items():
            if neuron_id in self.neurons:
                if neuron_id not in self.connections:
                    self.add_neuron(neuron_id)
                output = np.dot([input_value], [self.connections.get(neuron_id, {}).get(neuron_id, 0)])
                self.connections[neuron_id][neuron_id] = output
                yield output

    def get_connections(self, neuron_id):
        return self.connections.get(neuron_id, {})