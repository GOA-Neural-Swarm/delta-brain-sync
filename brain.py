import random

class NeuralNetwork:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses

    def compute(self, inputs):
        outputs = []
        for i, input_val in enumerate(inputs):
            neuron_outputs = []
            for neuron in self.neurons:
                if neuron["input"] == i:
                    if input_val > neuron["threshold"]:
                        neuron_outputs.append(1)
                    else:
                        neuron_outputs.append(0)
            outputs.append(max(neuron_outputs))

    def mutate(self):
        for synapse in self.synapses:
            if random.random() < 0.1:
                synapse["weight"] += random.gauss(0, 1)

    def reproduce(self):
        offspring = NeuralNetwork(self.neurons[:], self.synapses[:])
        for synapse in self.synapses:
            if random.random() < 0.5:
                synapse["weight"] += random.gauss(0, 1)
        return offspring

neurons = [{"input": 0, "threshold": 0.5}, {"input": 1, "threshold": 0.5}]
synapses = [{"weight": 0.5, "input": 0, "output": 0}, {"weight": 0.5, "input": 1, "output": 1}]

brain = NeuralNetwork(neurons, synapses)
brain.mutate()
brain.reproduce()