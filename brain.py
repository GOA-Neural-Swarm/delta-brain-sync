import numpy as np

class Brain:
    def __init__(self):
        self.connections = np.zeros((10, 10), dtype=int)
        self.learning_rate = 0.1

    def learn(self, input_data, output_data):
        for i in range(len(input_data)):
            self.connections[input_data[i]][output_data[i]] += self.learning_rate
        return self.connections

    def predict(self, input_data):
        output_data = np.zeros((1, 10), dtype=int)
        for i in range(len(input_data)):
            output_data[0][i] = np.argmax(np.dot(input_data[i], self.connections[i]))
        return output_data

    def evolve(self, new_connections):
        self.connections = new_connections
        return self.connections

brain = Brain()