import random

class Neuron:
    def __init__(self, id):
        self.id = id
        self.connections = []
        self.output = 0
        self.learning_rate = 0.1

    def fire(self, input_signal):
        self.output = self.learning_rate * input_signal
        return self.output

    def add_connection(self, neuron):
        self.connections.append(neuron)

class Brain:
    def __init__(self):
        self.neurons = [Neuron(i) for i in range(100)]

    def process(self):
        for neuron in self.neurons:
            input_signal = random.random()
            output = neuron.fire(input_signal)
            for connected_neuron in neuron.connections:
                connected_neuron.fire(output)

    def learn(self):
        for neuron in self.neurons:
            for connected_neuron in neuron.connections:
                neuron.learning_rate += random.uniform(-0.01, 0.01)

    def evolve(self):
        self.process()
        self.learn()
        for neuron in self.neurons:
            if neuron.learning_rate > 0.5:
                neuron.learning_rate = 0.5

brain = Brain()
for _ in range(1000):
    brain.evolve()
print("Brain Evolved!")