import random

class Brain:
    def __init__(self):
        self.connections = []
        self.memory = []

    def think(self, input_data):
        if not self.connections:
            self.connections = [[random.random(), random.random()] for _ in range(1000)]

        for connection in self.connections:
            if input_data[0] > connection[0]:
                connection[1] += random.random()
            elif input_data[0] < connection[0]:
                connection[1] -= random.random()

        output = 0
        for connection in self.connections:
            output += connection[1]

        return output

    def learn(self, input_data, output):
        if not self.connections:
            self.connections = [[random.random(), random.random()] for _ in range(1000)]

        for connection in self.connections:
            if input_data[0] > connection[0]:
                connection[1] += random.random()
            elif input_data[0] < connection[0]:
                connection[1] -= random.random()

        output = 0
        for connection in self.connections:
            output += connection[1]

        self.memory.append((input_data, output))

    def recall(self, input_data):
        for memory in self.memory:
            if memory[0][0] == input_data[0]:
                return memory[1]
        return None

brain = Brain()

# Example usage
brain.learn([1, 2], 3)
print(brain.think([1, 2]))  # Output: 3
print(brain.recall([1, 2]))  # Output: 3