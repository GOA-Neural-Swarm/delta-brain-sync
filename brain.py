class Brain:
    def __init__(self):
        self.synapses = {}

    def process(self, input_signal):
        output_signal = 0
        for synapse in self.synapses:
            weight = self.synapses[synapse]['weight']
            if synapse in input_signal:
                output_signal += input_signal[synapse] * weight
        return output_signal

    def train(self, input_signal, target_output):
        for synapse in self.synapses:
            weight = self.synapses[synapse]['weight']
            if synapse in input_signal:
                self.synapses[synapse]['weight'] += 0.1 * (target_output - input_signal[synapse] * weight)