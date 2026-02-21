import numpy as np

class RNAQT45:
    def __init__(self, sequence):
        self.sequence = sequence
        self.predictor = np.zeros((len(sequence), 1))

    def predict(self, index):
        if index == 0:
            return self.predictor[index]
        else:
            self.predictor[index] = np.sum(self.predictor[:index]) + self.sequence[index]
            return self.predictor[index]

    def recursive_upgrade(self, index):
        if index < len(self.sequence):
            self.predictor[index] = self.predict(index)
            return self.recursive_upgrade(index + 1)
        else:
            return self.predictor

sequence = "ATCGATCGATCG"
rqa = RNAQT45(sequence)
print(rqa.recursive_upgrade(0))