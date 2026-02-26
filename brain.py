import numpy as np
from typing import List

class Brain:
    def __init__(self, num_neurons: int):
        self.neurons: List[np.ndarray] = [np.zeros((num_neurons,)) for _ in range(num_neurons)]

    def process(self, inputs: List[float]) -> List[float]:
        for i, input_value in enumerate(inputs):
            self.neurons[i] += input_value
        return [np.tanh(neuron) for neuron in self.neurons]