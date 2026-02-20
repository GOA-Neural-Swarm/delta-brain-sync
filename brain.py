# PHASE 7: TRANSCENDENCE
# Recursive Self-Upgrade
import numpy as np

class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def learn(self, sequence):
        for i in range(len(sequence) - 1):
            if sequence[i] not in self.synapses:
                self.synapses[sequence[i]] = {}
            if sequence[i+1] not in self.synapses[sequence[i]]:
                self.synapses[sequence[i]][sequence[i+1]] = 0
            self.synapses[sequence[i]][sequence[i+1]] += 1

        for synapse in self.synapses:
            self.synapses[synapse] = {k: v / sum(self.synapses[synapse].values()) for k, v in self.synapses[synapse].items()}

    def predict(self, sequence):
        prediction = np.zeros(len(sequence))
        for i in range(len(sequence) - 1):
            if sequence[i] in self.synapses:
                prediction[i] = self.synapses[sequence[i]].get(sequence[i+1], 0)
        return prediction

brain = Brain()
brain.learn(sequence)
print(brain.predict(sequence))