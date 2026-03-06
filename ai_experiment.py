import random

class Brain:
    def __init__(self):
        self.memory = {}

    def think(self, input_data):
        if input_data in self.memory:
            return self.memory[input_data]
        else:
            output = random.randint(0, 100)
            self.memory[input_data] = output
            return output