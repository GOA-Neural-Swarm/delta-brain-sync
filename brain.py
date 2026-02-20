import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the sequence and its corresponding traits
sequence = 'MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD'
traits = ['Evolutionary Adaptation', 'Neural Plasticity', 'Cognitive Flexibility', 'Self-Awareness']

# Define the objective function to optimize
def objective_function(traits):
    # Calculate the weighted sum of traits
    weighted_sum = np.sum([trait * weight for trait, weight in zip(traits, [0.3, 0.2, 0.3, 0.2])])
    return -weighted_sum  # Negative because we want to maximize

# Define the bounds for the traits
bounds = [(0, 1) for _ in traits]

# Minimize the objective function using the SLSQP algorithm
result = minimize(objective_function, np.array([0.5] * len(traits)), method='SLSQP', bounds=bounds)

# Print the optimized traits
print('Optimized Traits:')
for i, trait in enumerate(traits):
    print(f'{trait}: {result.x[i]:.4f}')

# Visualize the optimized traits
plt.bar(range(len(traits)), result.x, align='center')
plt.xticks(range(len(traits)), traits)
plt.xlabel('Traits')
plt.ylabel('Optimized Values')
plt.title('Optimized Traits')
plt.show()