class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.signal = 0

    def fire(self):
        self.signal = 1 if self.signal >= self.threshold else 0

    def update(self, signal):
        self.signal += signal

    def get_signal(self):
        return self.signal