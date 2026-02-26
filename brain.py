class Brain:
    def __init__(self, num_neurons, num_synapses):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.neurons = [0] * num_neurons
        self.synapses = [[0] * num_synapses for _ in range(num_neurons)]

    def process(self, inputs):
        for i in range(self.num_neurons):
            self.neurons[i] = 0
            for j in range(self.num_synapses):
                self.neurons[i] += inputs[j] * self.synapses[i][j]
        return self.neurons