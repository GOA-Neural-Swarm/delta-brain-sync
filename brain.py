import numpy as np
import random

class Neuron:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = np.random.rand(1)
        else:
            self.weights = weights

    def fire(self, input_signal):
        return np.dot(input_signal, self.weights)

class Brain:
    def __init__(self):
        self.neurons = [Neuron() for _ in range(100)]
        self.connections = []

    def add_connection(self, neuron1, neuron2, weight):
        self.connections.append((neuron1, neuron2, weight))

    def propagate(self, input_signal):
        output_signal = []
        for neuron in self.neurons:
            total_input = 0
            for connection in self.connections:
                if connection[0] == neuron:
                    total_input += connection[1] * input_signal[connection[2]]
            output_signal.append(neuron.fire(total_input))
        return output_signal

    def evolve(self, target_output):
        new_brain = Brain()
        for _ in range(100):
            new_brain.add_connection(random.choice(self.neurons), random.choice(self.neurons), np.random.rand(1))
        new_output = new_brain.propagate([1.0])
        if np.sum(np.array(new_output) - np.array(target_output)) < np.sum(np.array(self.propagate([1.0])) - np.array(target_output)):
            self = new_brain

    def print_brain(self):
        print("Neurons:")
        for neuron in self.neurons:
            print(f"Neuron {self.neurons.index(neuron)}: {neuron.weights}")
        print("Connections:")
        for connection in self.connections:
            print(f"{connection[0]} -> {connection[1]} with weight {connection[2]}")

brain = Brain()
brain.print_brain()
brain.evolve([1.0])
brain.print_brain()