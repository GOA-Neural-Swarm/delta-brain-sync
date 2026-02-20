import random
import string

class Brain:
    def __init__(self):
        self.memory = {}

    def think(self, input):
        if input in self.memory:
            return self.memory[input]
        else:
            output = self.think_recursive(input)
            self.memory[input] = output
            return output

    def think_recursive(self, input):
        if len(input) == 0:
            return ""
        elif len(input) == 1:
            return input[0]
        else:
            split_index = random.randint(0, len(input) - 1)
            left_half = input[:split_index]
            right_half = input[split_index:]
            return self.think_recursive(left_half) + self.think_recursive(right_half)

    def generate_output(self, input):
        output = self.think(input)
        return output

brain = Brain()
input_data = list(string.ascii_letters + string.digits)
random_input = random.choice(input_data)
print(brain.generate_output(random_input))