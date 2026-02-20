class OptimizedBrain:
    def __init__(self):
        self.neural_network = {
            'input_layer': {'neurons': 5, 'activation_function':'sigmoid'},
            'hidden_layer': {'neurons': 10, 'activation_function':'relu'},
            'output_layer': {'neurons': 1, 'activation_function':'sigmoid'}
        }
        self.weights = {
            'input_to_hidden': [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            'hidden_to_output': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        }
        self.bias = {
            'hidden_layer': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'output_layer': [0.1]
        }

    def forward_pass(self, input_vector):
        hidden_layer = []
        for i in range(5):
            hidden_layer.append(self.neural_network['hidden_layer']['neurons'][i] * self.weights['input_to_hidden'][0][i] + self.bias['hidden_layer'][i])
        output = self.neural_network['output_layer']['neurons'][0] * self.weights['hidden_to_output'][0][0] + self.bias['output_layer'][0]
        return output

    def backward_pass(self, output_error):
        # Calculate error gradients
        hidden_layer_error = [0.1]
        for i in range(1, 10):
            hidden_layer_error.append(hidden_layer_error[i-1] * self.neural_network['hidden_layer']['neurons'][i-1])
        # Update weights and bias
        for i in range(5):
            self.weights['input_to_hidden'][0][i] += 0.01 * hidden_layer_error[0] * self.neural_network['hidden_layer']['neurons'][i]
        for i in range(10):
            self.weights['hidden_to_output'][0][i] += 0.01 * output_error * hidden_layer_error[i]
        self.bias['hidden_layer'][0] += 0.01 * hidden_layer_error[0]
        self.bias['output_layer'][0] += 0.01 * output_error

    def train(self, input_vectors, output_vectors):
        for input_vector, output_vector in zip(input_vectors, output_vectors):
            output = self.forward_pass(input_vector)
            error = output_vector - output
            self.backward_pass(error)