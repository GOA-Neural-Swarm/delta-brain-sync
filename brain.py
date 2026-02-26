class Brain:
    def __init__(self):
        self.neurons = {}

    def add_neuron(self, neuron_id, weights):
        self.neurons[neuron_id] = weights

    def process(self, inputs):
        import numpy as np
        import numba
        @numba.njit
        def compute_neuron_output(neuron_id, inputs):
            value = np.dot(np.array([weights.get(input_id, 0) for input_id in inputs]), np.array(list(inputs.values())))
            return value

        for neuron_id, weights in self.neurons.items():
            self.neurons[neuron_id] = compute_neuron_output(neuron_id, inputs)

    def get_output(self):
        outputs = {}
        for neuron_id, value in self.neurons.items():
            outputs[neuron_id] = value
        return outputs