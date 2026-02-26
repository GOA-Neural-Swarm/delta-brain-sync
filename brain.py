class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, id, func):
        self.neurons[id] = func

    def process(self, inputs):
        outputs = {}
        for neuron_id, neuron_func in self.neurons.items():
            output = neuron_func(inputs)
            outputs[neuron_id] = output
        return outputs

    def __call__(self, inputs):
        return self.process(inputs)