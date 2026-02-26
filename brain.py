import numba
@numba.jit(nopython=True)
def process_brain(brain, input_data):
    for neuron_id, inputs in brain.neurons.items():
        output = 0
        for input_neuron_id, weight in inputs['input_connections'].items():
            output += weight * input_data[input_neuron_id]
        brain.neurons[neuron_id]['output'] = output

brain = Brain()
brain.connect_neuron('neuron1', 'neuron2', 0.5)
brain.connect_neuron('neuron2', 'neuron3', 0.3)
input_data = {'neuron2': 1.0}
process_brain(brain, input_data)
print(brain.get_output('neuron1'))  # Output: 0.5