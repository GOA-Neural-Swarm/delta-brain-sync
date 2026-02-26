class Brain:
    def __init__(self):
        self.weights = {}
        self.biases = {}
        self.neurons = {}

    def add_neuron(self, id):
        self.neurons[id] = {'output': 0, 'weights': {}, 'bias': 0}

    def add_connection(self, neuron_id, target_id, weight):
        if neuron_id not in self.neurons:
            self.add_neuron(neuron_id)
        if target_id not in self.neurons:
            self.add_neuron(target_id)
        self.neurons[neuron_id]['weights'][target_id] = weight
        self.neurons[target_id]['weights'][neuron_id] = weight

    def propagate(self):
        for neuron_id in self.neurons:
            if neuron_id in self.neurons[neuron_id]['weights']:
                for target_id, weight in self.neurons[neuron_id]['weights'].items():
                    self.neurons[neuron_id]['output'] += self.neurons[target_id]['output'] * weight
                self.neurons[neuron_id]['output'] += self.neurons[neuron_id]['bias']
            else:
                self.neurons[neuron_id]['output'] = self.neurons[neuron_id]['bias']