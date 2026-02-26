class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def connect(self, neuron1, neuron2):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        if neuron2 not in self.synapses[neuron1]:
            self.synapses[neuron1][neuron2] = 0
        if neuron2 not in self.neurons:
            self.neurons[neuron2] = 0

    def fire(self, neuron):
        if neuron in self.synapses:
            for connected_neuron, strength in self.synapses[neuron].items():
                self.neurons[connected_neuron] += strength
        if neuron not in self.neurons:
            self.neurons[neuron] = 0

    def update(self):
        for neuron, value in self.neurons.items():
            if value > 1:
                self.fire(neuron)
                self.neurons[neuron] = 0