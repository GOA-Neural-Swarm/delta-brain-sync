import random

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.memory = []

    def learn(self):
        for gene in self.sequence:
            if random.random() < 0.1:
                self.memory.append(gene)
        return self.memory

    def recall(self):
        if len(self.memory) > 0:
            return random.choice(self.memory)
        else:
            return None

brain = Brain(PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)

for _ in range(100):
    brain.learn()
    print(brain.recall())