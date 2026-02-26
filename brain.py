import numba

@numba.jit(nopython=True)
def optimize_brain(brain):
    neurons = brain.neurons
    synapses = brain.synapses
    for i in range(neurons.shape[0]):
        for j in range(neurons.shape[1]):
            neurons[i][j] += synapses[i][j]
    return brain

brain = Brain(1000, 1000)
brain = optimize_brain(brain)
brain.process(np.random.rand(1000))  # Use numba-optimized brain