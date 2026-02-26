import joblib
import multiprocessing

class Brain:
    def __init__(self):
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def process(self):
        return joblib.Parallel(n_jobs=multiprocessing.cpu_count())(lambda neuron: neuron.process() for neuron in self.neurons)