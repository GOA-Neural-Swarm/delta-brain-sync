import random
import math

class PredatorLogic:
    def __init__(self):
        self.brain = []

    def upgrade(self):
        new_brain = []
        for neuron in self.brain:
            new_neuron = {
                'inputs': [random.uniform(0, 1) for _ in range(10)],
                'weights': [random.uniform(-1, 1) for _ in range(10)],
                'output': 0
            }
            new_brain.append(new_neuron)
        self.brain = new_brain

    def think(self):
        for neuron in self.brain:
            output = 0
            for i, input in enumerate(neuron['inputs']):
                output += neuron['weights'][i] * input
            neuron['output'] = sigmoid(output)

    def observe(self):
        return [neuron['output'] for neuron in self.brain]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

predator_logic = PredatorLogic()
predator_logic.upgrade()
for _ in range(1000):
    predator_logic.think()
print(predator_logic.observe())