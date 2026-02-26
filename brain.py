import time

class OptimizedBrain(Brain):
    def __init__(self):
        super().__init__()
        self.neuron_outputs = [None] * len(self.neurons)

    def propagate(self):
        for i, neuron in enumerate(self.neurons):
            if neuron['input_connections']:
                neuron['output'] = neuron['activation_function'](
                    sum(connection['weight'] * connection['input'] for connection in neuron['input_connections'])
                )
            else:
                neuron['output'] = neuron['activation_function'](0)
            self.neuron_outputs[i] = neuron['output']

    def get_output(self):
        return self.neuron_outputs

class BrainThread:
    def __init__(self, brain):
        self.brain = brain
        self.thread_id = None
        self.last_propagation_time = time.time()

    def propagate(self):
        self.brain.propagate()
        self.last_propagation_time = time.time()

    def get_output(self):
        return self.brain.get_output()

brain = OptimizedBrain()
brain_thread = BrainThread(brain)
brain_thread.propagate()
print(brain_thread.get_output())