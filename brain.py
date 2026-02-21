import numpy as np
import matplotlib.pyplot as plt

def RNAQT45PredatorLogic(DNA_sequence):
    # Load DNA sequence
    DNA = np.array(DNA_sequence)
    
    # Initialize logic matrix
    logic_matrix = np.zeros((len(DNA), len(DNA)))
    
    # Iterate through DNA sequence
    for i in range(len(DNA)):
        for j in range(len(DNA)):
            # Calculate Hamming distance
            hamming_distance = np.count_nonzero(DNA[i]!= DNA[j])
            
            # Calculate XOR operator
            xor_result = np.bitwise_xor(DNA[i], DNA[j])
            
            # Update logic matrix
            logic_matrix[i, j] = hamming_distance + xor_result
            
    # Normalize logic matrix
    logic_matrix = logic_matrix / np.max(logic_matrix)
    
    # Return logic matrix
    return logic_matrix

# Test DNA sequence
DNA_sequence = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T']
logic_matrix = RNAQT45PredatorLogic(DNA_sequence)

# Visualize logic matrix
plt.imshow(logic_matrix, cmap='hot', interpolation='nearest')
plt.show()