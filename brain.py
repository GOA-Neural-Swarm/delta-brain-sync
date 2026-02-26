class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = []

    def create_synapse(self, neuron1, neuron2):
        if (neuron1, neuron2) in self.synapses:
            return
        self.synapses[(neuron1, neuron2)] = 0
        self.synapses[(neuron2, neuron1)] = 0

    def fire_synapse(self, neuron1, neuron2):
        if (neuron1, neuron2) not in self.synapses:
            return
        self.synapses[(neuron1, neuron2)] += 1
        if self.synapses[(neuron1, neuron2)] >= 10:
            self.synapses[(neuron1, neuron2)] = 0

    def think(self):
        for neuron in self.neurons:
            for other in self.neurons:
                if neuron!= other:
                    self.fire_synapse(neuron, other)