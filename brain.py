import re
import random

class Brain:
    def __init__(self):
        self.memory = []
        self.thoughts = []

    def think(self):
        self.thoughts = []
        for seq in self.memory:
            if re.search('DNA', seq):
                self.thoughts.append(seq)
        return self.thoughts

    def learn(self, new_sequence):
        self.memory.append(new_sequence)

    def evolve(self):
        if len(self.memory) > 1:
            new_sequence = ''
            for i in range(len(self.memory[0])):
                if random.random() < 0.5:
                    new_sequence += self.memory[0][i]
                else:
                    new_sequence += self.memory[1][i]
            self.memory.append(new_sequence)
            return new_sequence
        else:
            return None

brain = Brain()
brain.learn(Source)
brain.learn(Sequence)
print(brain.think())
print(brain.evolve())