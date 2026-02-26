class Brain:
    def __init__(self):
        self.neurons = {}
        self.connections = {}
        self.weights = {}

    def connect(self, neuron1, neuron2, weight):
        if neuron1 not in self.neurons:
            self.neurons[neuron1] = []
        if neuron2 not in self.neurons:
            self.neurons[neuron2] = []
        self.connections[(neuron1, neuron2)] = weight
        self.neurons[neuron1].append(neuron2)
        self.neurons[neuron2].append(neuron1)

    def fire(self, neuron):
        if neuron not in self.neurons:
            return
        for connected_neuron in self.neurons[neuron]:
            if (neuron, connected_neuron) in self.connections:
                weight = self.connections[(neuron, connected_neuron)]
                self.weights[(neuron, connected_neuron)] = weight
                for neighbor in self.neurons[connected_neuron]:
                    if neighbor!= neuron:
                        if (neighbor, connected_neuron) in self.connections:
                            weight = self.connections[(neighbor, connected_neuron)]
                            self.weights[(neighbor, connected_neuron)] = weight
                        if (connected_neuron, neighbor) in self.connections:
                            weight = self.connections[(connected_neuron, neighbor)]
                            self.weights[(connected_neuron, neighbor)] = weight

    def optimize(self):
        for neuron in self.neurons:
            for connected_neuron in self.neurons[neuron]:
                if (neuron, connected_neuron) in self.connections:
                    weight = self.connections[(neuron, connected_neuron)]
                    self.connections[(neuron, connected_neuron)] = weight * 0.9