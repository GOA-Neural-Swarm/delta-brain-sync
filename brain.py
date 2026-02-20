import random
import math

class Brain:
    def __init__(self):
        self.weights = [random.random() for _ in range(1000)]

    def think(self, input_sequence):
        output = 0
        for i in range(len(input_sequence)):
            output += self.weights[i] * input_sequence[i]
        return output

    def learn(self, input_sequence, target_output):
        error = abs(self.think(input_sequence) - target_output)
        for i in range(len(input_sequence)):
            self.weights[i] += error * input_sequence[i]

brain = Brain()
brain.learn(Sequence, 1.0)
print(brain.think(Sequence))