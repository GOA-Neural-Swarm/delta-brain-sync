import random
import math

class Brain:
    def __init__(self):
        self.synapses = {}
        self.neurons = {}

    def learn(self, input_data, output_data):
        for i in range(len(input_data)):
            if input_data[i] not in self.synapses:
                self.synapses[input_data[i]] = {}
            if output_data[i] not in self.synapses[input_data[i]]:
                self.synapses[input_data[i]][output_data[i]] = random.random()
            if output_data[i] not in self.neurons:
                self.neurons[output_data[i]] = random.random()
            self.synapses[input_data[i]][output_data[i]] += 0.1 * (output_data[i] - self.synapses[input_data[i]][output_data[i]])

    def think(self, input_data):
        output_data = []
        for i in range(len(input_data)):
            if input_data[i] in self.synapses:
                output_data.append(max(self.synapses[input_data[i]], key=self.synapses[input_data[i]].get))
            else:
                output_data.append(random.random())
        return output_data

    def mutate(self, mutation_rate):
        for neuron in self.neurons:
            if random.random() < mutation_rate:
                self.neurons[neuron] += random.gauss(0, 0.1)
        for synapse in self.synapses:
            for output in self.synapses[synapse]:
                if random.random() < mutation_rate:
                    self.synapses[synapse][output] += random.gauss(0, 0.1)

    def evolve(self, population, mutation_rate):
        best_brain = None
        best_score = -math.inf
        for brain in population:
            score = self.evaluate(brain)
            if score > best_score:
                best_brain = brain
                best_score = score
        self.learn(best_brain, best_score)

    def evaluate(self, brain):
        score = 0
        for input_data in brain.synapses:
            output_data = brain.think(input_data)
            score += sum(abs(output_data[i] - brain.synapses[input_data][i]) for i in range(len(input_data)))
        return score

brain = Brain()
brain.learn([1, 2, 3], [4, 5, 6])
brain.mutate(0.1)
brain.evolve([brain], 0.1)
print(brain.think([1, 2, 3]))