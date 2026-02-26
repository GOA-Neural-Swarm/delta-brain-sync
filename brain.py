class Brain:
    def __init__(self):
        self.neurons = []
        self.connections = {}

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.connections[neuron.id] = []

    def connect_neurons(self, neuron1, neuron2, weight):
        self.connections[neuron1.id].append((neuron2, weight))
        self.connections[neuron2.id].append((neuron1, weight))

    def process(self):
        for neuron in self.neurons:
            neuron.update()
            for connected_neuron, weight in self.connections[neuron.id]:
                neuron.update_with_input(connected_neuron.output * weight)