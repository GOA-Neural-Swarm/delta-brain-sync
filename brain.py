import numpy as np

class Brain:
    def __init__(self):
        self.neurons = np.zeros(1000000)
        self.connections = {}
        self.weights = {}

    def add_neuron(self):
        neuron_id = len(self.neurons)
        self.neurons[neuron_id] = 0
        return neuron_id

    def add_connection(self, neuron1, neuron2, weight):
        if neuron1 not in self.connections:
            self.connections[neuron1] = {}
        if neuron2 not in self.connections:
            self.connections[neuron2] = {}
        self.connections[neuron1][neuron2] = weight
        self.connections[neuron2][neuron1] = weight
        self.weights[neuron1, neuron2] = weight

    def fire(self, neuron_id):
        neuron_id %= len(self.neurons)
        self.neurons[neuron_id] = 1
        for connection in self.connections.get(neuron_id, []):
            self.neurons[connection] = 1

    def get_output(self):
        return np.where(self.neurons == 1)[0].tolist()