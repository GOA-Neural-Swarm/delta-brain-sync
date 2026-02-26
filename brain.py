class Neuron:
    def __init__(self, id):
        self.id = id
        self.output = 0
        self.inputs = []

    def add_input(self, input_neuron):
        self.inputs.append(input_neuron)

    def update(self):
        self.output = self.calculate_output()
        for input_neuron in self.inputs:
            input_neuron.add_output(self)

    def calculate_output(self):
        # Optimized calculation for high-speed neural processing
        return sum(input_neuron.output for input_neuron in self.inputs)

class Connection:
    def __init__(self, neuron1, neuron2):
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.weight = 1

    def propagate(self):
        self.neuron2.add_input(self.neuron1)
        self.neuron1.output *= self.weight