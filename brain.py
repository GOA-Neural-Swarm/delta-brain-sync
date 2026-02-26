class Brain:
    def __init__(self):
        self.neurons = np.zeros(1000)
        self.weights = np.random.rand(1000, 1000)
        self.cache = np.zeros(1000)

    def process(self, inputs):
        self.neurons[:] = inputs
        self.neurons += np.dot(self.weights, self.neurons)
        self.cache[:] = self.neurons
        return self.neurons