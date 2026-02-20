import random

class Neuron:
    def __init__(self, synapses, activation_function):
        self.synapses = synapses
        self.activation_function = activation_function
        self.output = 0

    def fire(self, input_signal):
        self.output = self.activation_function(sum([synapse * input_signal for synapse in self.synapses]))

    def mutate(self):
        self.synapses = [synapse * random.uniform(0.9, 1.1) for synapse in self.synapses]

class Synapse:
    def __init__(self, strength):
        self.strength = strength

class Brain:
    def __init__(self, neurons):
        self.neurons = neurons

    def think(self, input_signal):
        for neuron in self.neurons:
            neuron.fire(input_signal)
        return [neuron.output for neuron in self.neurons]

    def evolve(self):
        for neuron in self.neurons:
            neuron.mutate()

# Define activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def ReLU(x):
    return max(0, x)

# Create brain with neurons and synapses
brain = Brain([Neuron([Synapse(0.5), Synapse(0.3)], sigmoid), Neuron([Synapse(0.8), Synapse(0.2)], ReLU)])

# Think and evolve the brain
for _ in range(100):
    input_signal = random.uniform(0, 1)
    output = brain.think(input_signal)
    print(output)
    brain.evolve()