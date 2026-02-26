import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}

    def learn(self, inputs, outputs):
        self.synapses = {i: {} for i in range(len(inputs))}
        for i in range(len(inputs)):
            for j in range(len(outputs)):
                if inputs[i] not in self.synapses[i]:
                    self.synapses[i][inputs[i]] = np.random.rand()
                self.synapses[i][inputs[i]] += outputs[j] * inputs[i]

    def predict(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            total = 0
            for j in range(len(inputs)):
                if inputs[j] in self.synapses[i]:
                    total += self.synapses[i][inputs[j]]
            outputs.append(total)
        return outputs

    def update(self, inputs, outputs):
        for i in range(len(inputs)):
            for j in range(len(outputs)):
                if inputs[i] in self.synapses[i]:
                    self.synapses[i][inputs[i]] += (outputs[j] - self.synapses[i][inputs[i]]) * inputs[i]