import random
import string

class Brain:
    def __init__(self):
        self.memory = {}

    def think(self, input_data):
        if input_data in self.memory:
            return self.memory[input_data]
        else:
            self.memory[input_data] = self.process(input_data)
            return self.memory[input_data]

    def process(self, input_data):
        if isinstance(input_data, str):
            return self.process_string(input_data)
        elif isinstance(input_data, int):
            return self.process_integer(input_data)
        else:
            return None

    def process_string(self, input_string):
        if random.random() < 0.5:
            return self.process_string(input_string + string.ascii_uppercase[0])
        else:
            return self.process_string(input_string + string.ascii_lowercase[0])

    def process_integer(self, input_integer):
        if random.random() < 0.5:
            return self.process_integer(input_integer + 1)
        else:
            return self.process_integer(input_integer - 1)

brain = Brain()