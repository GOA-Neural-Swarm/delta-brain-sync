import numba
from numba.experimental import jitclass

@jitclass
class Neuron:
    def __init__(self, func):
        self.func = func

    def __call__(self, inputs):
        return self.func(inputs)

brain = Brain()
brain.add_neuron('sigmoid', Neuron(sigmoid))
brain.add_neuron('tanh', Neuron(tanh))
brain.add_neuron('relu', Neuron(relu))

inputs = np.array([1, 2, 3])
outputs = brain(inputs)
print(outputs)

@numba.jit
def process(neuron_dict, inputs):
    outputs = {}
    for id, neuron in neuron_dict.items():
        outputs[id] = neuron(inputs)
    return outputs

brain = Brain()
brain.add_neuron('sigmoid', sigmoid)
brain.add_neuron('tanh', tanh)
brain.add_neuron('relu', relu)

inputs = np.array([1, 2, 3])
outputs = process(brain.neurons, inputs)
print(outputs)

@numba.jit
def __call__(self, inputs):
    return self.process(inputs)

brain = Brain()
brain.add_neuron('sigmoid', sigmoid)
brain.add_neuron('tanh', tanh)
brain.add_neuron('relu', relu)

inputs = np.array([1, 2, 3])
outputs = brain(inputs)
print(outputs)