brain = Brain()
brain.add_neuron('neuron1', Neuron('neuron1', [0.1, 0.2, 0.3]))
brain.add_neuron('neuron2', Neuron('neuron2', [0.4, 0.5, 0.6]))
brain.optimize()
brain.process([1, 2, 3])  # process input data and display output