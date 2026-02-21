import numpy as np
import pandas as pd

# Define the DNA sequence
sequence = "ATCGATCGATCGATCG"

# Convert the sequence to a numerical representation
sequence_numeric = np.array([4 if base == "A" else 3 if base == "T" else 2 if base == "C" else 1 for base in sequence])

# Perform Fourier Transform on the sequence
sequence_fft = np.fft.fft(sequence_numeric)

# Filter the FFT to extract relevant frequency components
filtered_fft = np.fft.ifft(sequence_fft * np.hamming(len(sequence_fft)))

# Convert the filtered FFT back to a DNA sequence
filtered_sequence = np.array([chr(int(base)) for base in filtered_fft.real])

print(filtered_sequence)