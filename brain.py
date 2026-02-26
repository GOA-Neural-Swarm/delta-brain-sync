class Brain:
    def __init__(self):
        self.neurons = {}
        self.synapses = {}

    def process(self, input_data):
        for neuron_id, weights in self.synapses.items():
            output = 0
            for input_id, weight in weights.items():
                output += self.neurons[input_id] * weight
            self.neurons[neuron_id] = sigmoid(output)
        return self.neurons

    def learn(self, input_data, target_output):
        for neuron_id, weights in self.synapses.items():
            output = 0
            for input_id, weight in weights.items():
                output += self.neurons[input_id] * weight
            error = target_output - self.neurons[neuron_id]
            for input_id, weight in weights.items():
                self.synapses[neuron_id][input_id] += 0.1 * error * self.neurons[input_id]
            self.neurons[neuron_id] = sigmoid(output)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def initialize(self, num_neurons, num_synapses):
        self.neurons = {i: random.random() for i in range(num_neurons)}
        self.synapses = {i: {j: random.random() for j in range(num_synapses)} for i in range(num_synapses)}