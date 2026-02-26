class Brain:
    def __init__(self):
        self.neurons = {}
        self.connections = []

    def add_neuron(self, id, value):
        self.neurons[id] = value

    def add_connection(self, from_neuron, to_neuron, weight):
        self.connections.append((from_neuron, to_neuron, weight))

    def process(self):
        for from_neuron, to_neuron, weight in self.connections:
            self.neurons[to_neuron] += self.neurons[from_neuron] * weight