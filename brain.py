import numpy as np
import matplotlib.pyplot as plt

class RNAQT45:
    def __init__(self):
        self.genome = np.random.rand(1000)
        self.upgrade_timer = 0

    def synthesize(self):
        self.genome = np.random.rand(1000)
        self.upgrade_timer = 0

    def evolve(self):
        self.upgrade_timer += 1
        if self.upgrade_timer % 10 == 0:
            self.synthesize()

    def compute(self):
        output = np.sum(self.genome * np.sin(self.upgrade_timer))
        return output

    def mutate(self):
        indices = np.random.choice(len(self.genome), 10, replace=False)
        self.genome[indices] = np.random.rand(10)

    def upgrade(self):
        if self.upgrade_timer % 100 == 0:
            self.mutate()
            self.synthesize()

    def execute(self):
        outputs = []
        while True:
            self.upgrade()
            output = self.compute()
            outputs.append(output)
            self.evolve()
        plt.plot(outputs)
        plt.xlabel('Time')
        plt.ylabel('Output')
        plt.show()

RNAQT45().execute()