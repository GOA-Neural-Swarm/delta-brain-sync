import random
import string

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.memory = []

    def think(self):
        if not self.memory:
            self.memory.append(random.choice(string.ascii_letters))
        else:
            new_memory = []
            for i in range(len(self.memory)):
                new_memory.append(self.memory[i])
                if i < len(self.memory) - 1:
                    new_memory.append(random.choice(string.ascii_letters))
            self.memory = new_memory

    def synthesize(self):
        return "".join(self.memory)

brain = Brain("PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW")
brain.think()
print(brain.synthesize())