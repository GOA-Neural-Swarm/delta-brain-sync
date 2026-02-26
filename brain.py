import numpy as np
from brain import Brain

brain = Brain(3, 1)

# Test the optimized Brain class
inputs = np.array([[0, 0, 1]])
output = brain.predict(inputs)
print(output)