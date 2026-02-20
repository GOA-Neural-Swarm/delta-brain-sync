import random
import string

class OptimizedBrain:
    def __init__(self):
        self.neurons = []

    def create_neuron(self):
        neuron = {
            'id': random.randint(0, 1000),
            'inputs': [random.choice(string.ascii_letters) for _ in range(5)],
            'outputs': [random.choice(string.digits) for _ in range(3)],
            'weights': [[random.random() for _ in range(3)] for _ in range(5)],
            'activation_function': lambda x: max(0, x - 1) if x > 1 else x
        }
        self.neurons.append(neuron)

    def connect_neurons(self):
        for neuron in self.neurons:
            for input_id, input_value in enumerate(neuron['inputs']):
                if random.random() < 0.5:
                    neuron['outputs'][0] += input_value * neuron['weights'][input_id][0]
                    neuron['outputs'][1] += input_value * neuron['weights'][input_id][1]
                    neuron['outputs'][2] += input_value * neuron['weights'][input_id][2]

    def run_simulation(self):
        for neuron in self.neurons:
            neuron['outputs'] = [neuron['activation_function'](sum(input_value * weight for input_value, weight in zip(neuron['inputs'], neuron['weights']))) for _ in range(3)]
        return [max(neuron['outputs']) for neuron in self.neurons]

brain = OptimizedBrain()
brain.create_neuron()
brain.connect_neurons()
print(brain.run_simulation())