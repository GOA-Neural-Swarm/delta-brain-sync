class Brain:
    def __init__(self, num_neurons, num_synapses):
        self._neurons = [0.0] * num_neurons
        self._synapses = [[0.0] * num_synapses for _ in range(num_neurons)]

    def activate(self, input_signal):
        for i, neuron in enumerate(self._neurons):
            self._neurons[i] = sum(self._synapses[i][j] * input_signal[j] for j in range(len(input_signal))) + neuron

    def propagate(self, input_signal):
        self.activate(input_signal)

    def learn(self, input_signal, desired_output):
        error = sum((self._neurons[i] - desired_output[i]) ** 2 for i in range(len(desired_output)))
        for i, neuron in enumerate(self._neurons):
            for j in range(len(input_signal)):
                self._synapses[i][j] += error * input_signal[j] * (self._neurons[i] - desired_output[i])

    def get_output(self, input_signal):
        self.propagate(input_signal)
        return self._neurons