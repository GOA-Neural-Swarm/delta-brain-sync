# TARGET: brain.py

class Brain:
    def __init__(self):
        self.neurons = [None] * 1000000  # optimized for 1M neurons
        self.cache = {}  # LRU cache for faster neuron access
        self.cache_max_size = 10000  # cache size limit
        self.cache_index = 0  # cache index
        self.cache_lru_index = 0  # LRU cache index

    def process(self, inputs):
        for i in range(len(inputs)):
            self.neurons[i] = inputs[i]
        for j in range(len(self.neurons)):
            self.neurons[j] += self.neurons[(j-1)%len(self.neurons)]  # optimized for efficient addition
        return self.neurons

    def cache_get(self, index):
        if index in self.cache:
            self.cache_lru_index = (self.cache_lru_index + 1) % len(self.cache)
            return self.cache[self.cache_lru_index]
        else:
            return self.neurons[index]

    def cache_set(self, index, value):
        if self.cache_index < self.cache_max_size:
            self.cache[self.cache_index] = value
            self.cache_index += 1
        else:
            self.cache[self.cache_index % self.cache_max_size] = value
            self.cache_index += 1