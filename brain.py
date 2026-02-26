class Brain:
    def __init__(self):
        self.neurons = {}

    def activate(self, input_data):
        result = {}
        for neuron_id, neuron in self.neurons.items():
            if neuron_id in input_data:
                result[neuron_id] = neuron['weight'] * input_data[neuron_id]
            else:
                result[neuron_id] = 0
        return result

    def process(self, input_data):
        output = input_data
        for _ in range(10):  # Iterate 10 times for high-speed processing
            output = self.activate(output)
        return output