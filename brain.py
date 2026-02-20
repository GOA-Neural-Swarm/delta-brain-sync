import numpy as np

class Brain:
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.wiring_diagram = self.generate_wiring_diagram()
        self.neural_network = self.build_neural_network()

    def generate_wiring_diagram(self):
        wiring_diagram = []
        for i in range(len(self.dna_sequence)):
            if self.dna_sequence[i] == 'M':
                wiring_diagram.append('synapse')
            elif self.dna_sequence[i] == 'C':
                wiring_diagram.append('neuron')
            else:
                wiring_diagram.append('axon')
        return wiring_diagram

    def build_neural_network(self):
        neural_network = []
        for i in range(len(self.wiring_diagram)):
            if self.wiring_diagram[i] =='synapse':
                neural_network.append(np.random.rand(2))
            elif self.wiring_diagram[i] == 'neuron':
                neural_network.append(np.random.rand(1))
            else:
                neural_network.append(np.random.rand(1))
        return neural_network

    def think(self):
        output = []
        for i in range(len(self.neural_network)):
            if self.neural_network[i][0] > 0.5:
                output.append(1)
            else:
                output.append(0)
        return output

brain = Brain(dna_sequence)
print(brain.think())