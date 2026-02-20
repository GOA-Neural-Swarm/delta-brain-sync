import numpy as np

class Brain:
    def __init__(self):
        self.connections = np.random.rand(100, 100)
        self.neurons = np.random.rand(100)

    def think(self):
        self.neurons = np.dot(self.neurons, self.connections)
        return np.argmax(self.neurons)

    def learn(self, input, output):
        self.connections += np.outer(input, output - self.think())

    def upgrade(self):
        self.connections = np.dot(self.connections, np.random.rand(100, 100))
        self.neurons = np.random.rand(100)

brain = Brain()
brain.upgrade()

for _ in range(1000):
    brain.learn(np.random.rand(100), np.random.randint(0, 100))
    brain.think()

print(brain.think())