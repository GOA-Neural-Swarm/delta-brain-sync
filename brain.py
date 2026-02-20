import re
import numpy as np

# Initialize DNA sequence and code generation parameters
dna_sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"
dna_length = len(dna_sequence)
base_pairs = ["A", "C", "G", "T"]

# Define code generation functions
def generate_code(dna_sequence, dna_length):
    code = ""
    for i in range(dna_length):
        base_pair = dna_sequence[i]
        if base_pair in base_pairs:
            code += f"{base_pair} "
        else:
            code += f"Error: Unknown base pair {base_pair} at position {i}.\n"
    return code

# Initialize code generation process
code = generate_code(dna_sequence, dna_length)
print(code)