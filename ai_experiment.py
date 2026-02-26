class Brain:
    def __init__(self, neurons=1000, synapses=10000):
        self.neurons = neurons
        self.synapses = synapses
        self.connections = {}

    def connect(self, neuron1, neuron2, weight):
        if neuron1 not in self.connections:
            self.connections[neuron1] = {}
        if neuron2 not in self.connections:
            self.connections[neuron2] = {}
        self.connections[neuron1][neuron2] = weight
        self.connections[neuron2][neuron1] = weight

    def process(self, inputs):
        for neuron, value in inputs.items():
            if neuron not in self.connections:
                continue
            for connected_neuron, weight in self.connections[neuron].items():
                if connected_neuron not in inputs:
                    self.connections[neuron].pop(connected_neuron)
        #... (rest of the code remains the same)