class Brain:
    def __init__(self):
        self.synapses = {}  # dict for fast lookup

    def connect(self, neuron1, neuron2, weight):
        self.synapses[(neuron1, neuron2)] = weight

    def fire(self, neuron):
        for synapse, weight in self.synapses.items():
            if synapse[0] == neuron:
                self.synapses[synapse] = weight + 1
                break

    def propagate(self, neuron):
        for synapse, weight in self.synapses.items():
            if synapse[1] == neuron:
                self.synapses[synapse] = weight + 1
                break

    def get_weight(self, neuron1, neuron2):
        return self.synapses.get((neuron1, neuron2), 0)