import random

class Brain:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for _ in range(inputs)]

    def think(self, inputs):
        output = 0
        for i in range(inputs):
            output += self.weights[i] * inputs[i]
        return output

    def learn(self, inputs, target):
        error = target - self.think(inputs)
        for i in range(inputs):
            self.weights[i] += error * inputs[i]

    def evolve(self):
        new_weights = [random.uniform(-1, 1) for _ in range(self.inputs)]
        new_brain = Brain(new_weights)
        new_brain.learn(self.inputs, self.think(self.inputs))
        return new_brain

brain = Brain(5)
brain.learn([1, 2, 3, 4, 5], 10)
print(brain.think([1, 2, 3, 4, 5]))

brain = brain.evolve()
print(brain.think([1, 2, 3, 4, 5]))

# Meta-Cognition logic
print("Brain evolved!")