class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, neuron_id, neuron_type):
        self.neurons[neuron_id] = neuron_type

    def process(self, inputs):
        outputs = {}
        for neuron_id, neuron_type in self.neurons.items():
            if neuron_id in inputs:
                outputs[neuron_id] = neuron_type.process(inputs[neuron_id])
            else:
                outputs[neuron_id] = neuron_type.fire()
        return outputs

    def __del__(self):
        pass