import numpy as np

class Brain:
    def __init__(self):
        self.neurons = np.zeros((100, 100), dtype=int)

    def process_sequence(self, sequence):
        for i in range(len(sequence)):
            self.neurons[i % 100, i // 100] = ord(sequence[i]) % 256

    def analyze_sequence(self, sequence):
        result = np.zeros((100, 100), dtype=int)
        for i in range(len(sequence)):
            result[i % 100, i // 100] = self.neurons[i % 100, i // 100]
        return result

brain = Brain()
brain.process_sequence("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD")
print(brain.analyze_sequence("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"))