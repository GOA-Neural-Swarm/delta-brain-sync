class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, name, value):
        self.neurons[name] = value

    def fire_synapse(self, source, target):
        if source in self.neurons and target in self.neurons:
            self.neurons[target] += self.neurons[source]
        else:
            print("Invalid neuron names")

    def update_neurons(self):
        for neuron in self.neurons:
            self.neurons[neuron] *= 0.95