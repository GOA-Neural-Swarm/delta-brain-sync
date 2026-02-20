import random

class Brain:
    def __init__(self):
        self.neural_network = []
        self.memory = []

    def learn(self, sequence):
        for i in range(len(sequence)):
            if sequence[i] == 'A':
                self.neural_network.append(random.randint(0, 1))
            elif sequence[i] == 'C':
                self.neural_network.append(random.randint(0, 1))
            elif sequence[i] == 'G':
                self.neural_network.append(random.randint(0, 1))
            elif sequence[i] == 'T':
                self.neural_network.append(random.randint(0, 1))

    def recall(self):
        return self.neural_network

    def adapt(self):
        for i in range(len(self.neural_network)):
            if self.neural_network[i] == 0:
                self.neural_network[i] = 1
            else:
                self.neural_network[i] = 0

    def optimize(self):
        self.adapt()
        self.learn(MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD)

brain = Brain()
brain.learn(MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD)
print(brain.recall())
brain.optimize()
print(brain.recall())