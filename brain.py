
brain = Brain()
neuron1 = Neuron(1)
neuron2 = Neuron(2)
synapse = Synapse(neuron1, neuron2)
brain.add_neuron(neuron1)
brain.add_neuron(neuron2)
brain.add_synapse(synapse)
brain.process()