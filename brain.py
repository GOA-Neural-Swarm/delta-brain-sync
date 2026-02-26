import time
import random

class DistributedBrain(Brain):
    def __init__(self):
        super().__init__()
        self._servers = []

    def _run(self, neuron):
        output = [n for n in self.neurons[neuron]['output']]
        return output

    def run(self):
        start_time = time.time()
        with distributed.Client() as client:
            self._servers = client.submit(self._run, list(self.neurons.keys()))
            results = client.gather(self._servers)
        end_time = time.time()
        print(f"Run time: {end_time - start_time:.6f} seconds")
        return [n for n in self.neurons.values() if n['output']]

import distributed
brain = DistributedBrain()
brain.add_neuron('A')
brain.add_neuron('B')
brain.connect_neurons('A', 'B')
brain.run()