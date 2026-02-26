import numpy as np

class Brain:
    def __init__(self, layers):
        self.layers = layers
        self.synapses = {}
        for i in range(len(layers) - 1):
            self.synapses[(i, i + 1)] = np.random.rand(layers[i], layers[i + 1])

    def forward_pass(self, inputs):
        inputs = np.array(inputs)
        outputs = inputs
        for i, layer in enumerate(self.layers[1:]):
            outputs = np.dot(outputs, self.synapses[(i, i + 1)])
            outputs = self._activation(outputs)
        return outputs

    def _activation(self, x):
        return np.tanh(x)

    def train(self, inputs, targets):
        for i in range(len(self.layers) - 1):
            layer_in = inputs
            layer_out = self.synapses[(i, i + 1)]
            layer_targets = targets
            for j in range(len(layer_out)):
                for k in range(len(layer_in)):
                    error = layer_targets[j] - layer_out[j]
                    self.synapses[(i, i + 1)][k, j] += 0.1 * error * layer_in[k]