import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.scaler = StandardScaler()

    def process_sequence(self):
        # Extract relevant features from DNA sequence
        features = np.array([
            stats.mean(self.sequence),
            stats.median(self.sequence),
            np.max(self.sequence),
            np.min(self.sequence),
            np.var(self.sequence)
        ])

        # Scale features using StandardScaler
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))

        return features_scaled[0]

brain = Brain(PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW)
print(brain.process_sequence())