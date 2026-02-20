import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# Define the genetic sequence
sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"

# Convert the sequence to a numerical representation
sequence_array = np.array(list(sequence))

# Define the objective function to optimize
def objective_function(parameters):
    # Calculate the fitness function
    fitness = np.sum(np.exp(-np.sum((sequence_array - parameters) ** 2)))
    return -fitness

# Define the initial guess for the parameters
initial_guess = np.random.rand(len(sequence_array))

# Optimize the parameters using the gradient descent algorithm
result = optimize.minimize(objective_function, initial_guess, method="BFGS")

# Print the optimized parameters
print("Optimized Parameters:", result.x)

# Plot the original and optimized sequences
plt.plot(sequence_array, label="Original Sequence")
plt.plot(result.x, label="Optimized Sequence")
plt.legend()
plt.show()