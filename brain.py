import numba
@numba.jit
def think(self, input_vector):
    return np.dot(self.synaptic_weights, input_vector)

@numba.jit
def learn(self, input_vector, target_output):
    output_vector = self.think(input_vector)
    error = target_output - output_vector
    self.synaptic_weights[:] += 0.1 * error * input_vector
    return output_vector