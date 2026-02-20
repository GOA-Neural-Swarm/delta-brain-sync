import random

class Brain:
    def __init__(self):
        self.weights = [random.random() for _ in range(100)]
        self.biases = [random.random() for _ in range(10)]
        self.synapses = [[random.random() for _ in range(10)] for _ in range(100)]

    def think(self, input):
        output = []
        for i in range(10):
            sum = 0
            for j in range(100):
                sum += input[j] * self.synapses[j][i]
            sum += self.biases[i]
            output.append(sigmoid(sum))
        return output

    def learn(self, input, target):
        delta = [0] * 10
        for i in range(10):
            sum = 0
            for j in range(100):
                sum += input[j] * self.synapses[j][i]
            sum += self.biases[i]
            error = target[i] - sigmoid(sum)
            delta[i] = error * (1 - sigmoid(sum)) * sigmoid(sum)
            self.biases[i] += 0.1 * delta[i]
            for j in range(100):
                self.synapses[j][i] += 0.1 * delta[i] * input[j]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def exp(self, x):
        return math.exp(x)

brain = Brain()
brain.learn([1, 0, 1, 1, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1, 1, 0, 1])
print(brain.think([1, 0, 1, 1, 1, 0, 1, 1, 0, 1]))