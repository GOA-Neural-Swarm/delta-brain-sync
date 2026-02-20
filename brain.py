import numpy as np

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.neurons = {}
        self.connections = {}

    def evolve(self):
        for i in range(len(self.sequence)):
            if self.sequence[i] == 'M':
                self.neurons[i] = {'type': 'input'}
            elif self.sequence[i] == 'C':
                self.neurons[i] = {'type': 'hidden'}
            elif self.sequence[i] == 'P':
                self.neurons[i] = {'type': 'output'}
            else:
                self.neurons[i] = {'type': 'unknown'}

        for i in range(len(self.sequence) - 1):
            if self.sequence[i] == 'M' and self.sequence[i + 1] == 'C':
                self.connections[i] = {'weight': np.random.uniform(-1, 1)}
            elif self.sequence[i] == 'C' and self.sequence[i + 1] == 'P':
                self.connections[i] = {'weight': np.random.uniform(-1, 1)}
            else:
                self.connections[i] = {'weight': 0}

    def think(self, input_data):
        for i in range(len(input_data)):
            if self.neurons[i]['type'] == 'input':
                self.neurons[i]['value'] = input_data[i]
            elif self.neurons[i]['type'] == 'hidden':
                if i > 0:
                    self.neurons[i]['value'] = self.connections[i-1]['weight'] * self.neurons[i-1]['value']
                else:
                    self.neurons[i]['value'] = 0
            elif self.neurons[i]['type'] == 'output':
                if i > 0:
                    self.neurons[i]['value'] = self.connections[i-1]['weight'] * self.neurons[i-1]['value']
                else:
                    self.neurons[i]['value'] = 0

        return self.neurons[-1]['value']

brain = Brain('MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD')
brain.evolve()
print(brain.think([1, 2, 3, 4, 5]))