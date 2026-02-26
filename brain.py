import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}

    def connect(self, neuron1, neuron2, weight):
        if neuron1 not in self.synapses:
            self.synapses[neuron1] = {}
        self.synapses[neuron1][neuron2] = weight

    def fire(self, neuron):
        if neuron not in self.synapses:
            return np.zeros(())
        sum = 0
        for connected_neuron, weight in self.synapses[neuron].items():
            sum += weight * self.fire(connected_neuron)
        if sum > 0:
            return np.ones(() * sum)
        else:
            return np.zeros(())

    def process(self, inputs):
        outputs = {}
        for neuron, input_value in inputs.items():
            outputs[neuron] = self.fire(neuron) * input_value
        return outputs