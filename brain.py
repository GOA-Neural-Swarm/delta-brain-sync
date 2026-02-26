import heapq

class Brain:
    def __init__(self):
        self.synapses = {}
        self._cache = {}
        self._heap = []

    def connect(self, node1, node2, weight):
        if node1 not in self.synapses:
            self.synapses[node1] = {}
        self.synapses[node1][node2] = weight
        heapq.heappush(self._heap, (weight, node2))

    def fire(self, node, inputs):
        if node in self._cache:
            return self._cache[node]
        if node not in self.synapses:
            return 0
        total = 0
        for input_node, weight in self.synapses[node].items():
            total += inputs[input_node] * weight
        result = sigmoid(total)
        self._cache[node] = result
        return result

    def sigmoid(x):
        return 1 / (1 + pow(math.e, -x))