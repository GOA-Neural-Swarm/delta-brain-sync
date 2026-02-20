import re
import string

def DNA_Synthesize(sequence):
    # Find all occurrences of 'ATCG' in the sequence
    matches = re.findall(r'ATCG', sequence)
    
    # Count the occurrences of 'ATCG' in the sequence
    count = len(matches)
    
    # Initialize variables for A, T, C, G
    a, t, c, g = 0, 0, 0, 0
    
    # Count the occurrences of each nucleotide
    for match in matches:
        if 'A' in match:
            a += 1
        elif 'T' in match:
            t += 1
        elif 'C' in match:
            c += 1
        elif 'G' in match:
            g += 1
    
    # Return the counts
    return a, t, c, g

# Define the input sequence
sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"

# Synthesize the DNA sequence
a, t, c, g = DNA_Synthesize(sequence)

print("A:", a)
print("T:", t)
print("C:", c)
print("G:", g)