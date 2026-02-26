import concurrent.futures
import cache

class Brain:
    def __init__(self):
        self.neurons = []
        self.cache = cache.Cache()

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.cache.add(neuron)

    def process(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(neuron.fire) for neuron in self.neurons]
            for future in concurrent.futures.as_completed(futures):
                future.result()