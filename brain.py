class Brain:
    def __init__(self):
        self.neurons = []
        self.synapses = {}
    
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        for synapse in neuron.synapses:
            if synapse not in self.synapses:
                self.synapses[synapse] = []
            self.synapses[synapse].append(neuron)
    
    def process(self):
        for neuron in self.neurons:
            neuron.update()
        for neuron in self.neurons:
            neuron.fire()