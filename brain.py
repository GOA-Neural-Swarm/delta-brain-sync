import random

class PredatorLogic:
    def __init__(self):
        self.brain = []

    def upgrade(self):
        new_neuron = random.randint(0, 1000)
        self.brain.append(new_neuron)

    def process(self, input_data):
        output = 0
        for neuron in self.brain:
            output += neuron * input_data
        return output

    def mutate(self, mutation_rate):
        for i, neuron in enumerate(self.brain):
            if random.random() < mutation_rate:
                self.brain[i] += random.randint(-100, 100)

    def evolve(self, generations, mutation_rate):
        for _ in range(generations):
            self.upgrade()
            self.mutate(mutation_rate)

    def think(self, input_data):
        return self.process(input_data)

predator = PredatorLogic()
predator.evolve(100, 0.1)
print(predator.think(5))