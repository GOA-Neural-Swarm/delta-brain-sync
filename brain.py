import numpy as np

class Brain:
    def __init__(self):
        self.genome = np.array(Source, dtype=np.uint8)
        self.neurons = {}

    def initialize_neurons(self):
        for i in range(len(self.genome)):
            if self.genome[i] == 1:
                self.neurons[i] = {'type': 'input', 'value': 0}
            elif self.genome[i] == 2:
                self.neurons[i] = {'type': 'hidden', 'value': 0}
            elif self.genome[i] == 3:
                self.neurons[i] = {'type': 'output', 'value': 0}

    def calculate_output(self):
        for neuron in self.neurons.values():
            if neuron['type'] == 'input':
                neuron['value'] = neuron['value']
            elif neuron['type'] == 'hidden':
                neuron['value'] = np.tanh(np.sum([self.neurons[i]['value'] for i in range(len(self.genome)) if self.genome[i] == 2]))
            elif neuron['type'] == 'output':
                neuron['value'] = np.tanh(np.sum([self.neurons[i]['value'] for i in range(len(self.genome)) if self.genome[i] == 3]))

    def optimize(self):
        for neuron in self.neurons.values():
            if neuron['type'] == 'input':
                neuron['value'] = neuron['value']
            elif neuron['type'] == 'hidden':
                neuron['value'] = np.tanh(np.sum([self.neurons[i]['value'] for i in range(len(self.genome)) if self.genome[i] == 2]))
            elif neuron['type'] == 'output':
                neuron['value'] = np.tanh(np.sum([self.neurons[i]['value'] for i in range(len(self.genome)) if self.genome[i] == 3]))
                if neuron['value'] > 0.5:
                    neuron['value'] = 1
                else:
                    neuron['value'] = 0

    def evolve(self):
        mutation_rate = 0.1
        for i in range(len(self.genome)):
            if np.random.rand() < mutation_rate:
                if self.genome[i] == 1:
                    self.genome[i] = 2
                elif self.genome[i] == 2:
                    self.genome[i] = 3
                elif self.genome[i] == 3:
                    self.genome[i] = 1

brain = Brain()
brain.initialize_neurons()
brain.optimize()
brain.evolve()