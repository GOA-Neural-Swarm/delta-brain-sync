import numpy as np

class Brain:
    def __init__(self, inputs, outputs, layers):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = [np.zeros((i, o)) for i, o in zip([inputs]+[l[1] for l in layers]+[outputs], [outputs]+[l[2] for l in layers]+[inputs])]
        self.weights = [[np.random.rand(i, o) for o in [l[1] for l in layers]+[outputs]] for i in [inputs]+[l[1] for l in layers]+[outputs]]
        self.biases = [[np.zeros((i, 1)) for i in [l[1] for l in layers]+[outputs]] for _ in range(len(layers)+1)]

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            layer_inputs = [np.append(self.layers[0], inputs)]
            for i, (in_layer, out_layer, _) in enumerate(self.layers[1:]):
                layer_inputs.append(np.dot(in_layer, self.weights[i+1].T)+self.biases[i+1])
                layer_inputs[i+1] = self.sigmoid(layer_inputs[i+1])
            error = np.sum((layer_inputs[-1]-outputs)**2)
            dweights = [np.dot(layer_inputs[i+1].T, layer_inputs[i+1]-outputs)]
            for i, (in_layer, out_layer, _) in enumerate(self.layers[:-1][::-1]):
                dweights[i] = np.dot(dweights[i+1], out_layer) + dweights[i]
                layer_inputs[i+1] = self.sigmoid_derivative(layer_inputs[i+1])
            self.weights = [w - 0.1 * dw for w, dw in zip(self.weights, dweights)]
            self.biases = [b - 0.1 * np.sum((dw > 0) * (dw < 1), axis=0, keepdims=True) for b, dw in zip(self.biases, dweights)]

    def think(self, inputs):
        layer_inputs = [np.append(self.layers[0], inputs)]
        for i, (in_layer, out_layer, _) in enumerate(self.layers[1:]):
            layer_inputs.append(np.dot(in_layer, self.weights[i+1].T)+self.biases[i+1])
            layer_inputs[i+1] = self.sigmoid(layer_inputs[i+1])
        return layer_inputs[-1]