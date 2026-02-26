class Brain:
    def __init__(self):
        self.neurons = [{} for _ in range(10000)]  # Pre-allocate neuron dict array
        self.connections = [{} for _ in range(10000)]  # Pre-allocate connection dict array

    def process(self, input_data):
        for neuron_id, inputs in enumerate(input_data):
            if inputs:  # Non-empty input list
                for input_id, weight in inputs:
                    self.connections[neuron_id][input_id] += weight
            if not self.connections[neuron_id]:  # No connections, reset neuron
                self.connections[neuron_id] = {}

    def update(self):
        for neuron_id, connections in enumerate(self.connections):
            if connections:
                total_weight = sum(connections.values())
                self.neurons[neuron_id] = total_weight  # Update neuron state
            else:
                self.neurons[neuron_id] = 0  # Reset neuron state