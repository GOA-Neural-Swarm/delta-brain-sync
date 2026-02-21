import numpy as np
import pandas as pd
from scipy.stats import norm

class TelefoxX:
    def __init__(self):
        self.upgrade_data = np.array([])

    def upgrade(self, data):
        if not self.upgrade_data.any():
            self.upgrade_data = np.array([data])
        else:
            self.upgrade_data = np.concatenate((self.upgrade_data, np.array([data])))

    def analyze(self):
        mean = np.mean(self.upgrade_data)
        std = np.std(self.upgrade_data)
        return norm.pdf(mean, mean, std)

telefox = TelefoxX()
telefox.upgrade(np.array([1, 2, 3, 4, 5]))
print(telefox.analyze())