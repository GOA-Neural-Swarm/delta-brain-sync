import random
import string

def generate_neon_sequence(length):
    sequence = ''
    for _ in range(length):
        nucleotide = random.choice(['A', 'C', 'G', 'T'])
        sequence += nucleotide
    return sequence

def neon_analysis(sequence):
    frequency = {}
    for nucleotide in sequence:
        if nucleotide in frequency:
            frequency[nucleotide] += 1
        else:
            frequency[nucleotide] = 1
    return frequency

def optimize_sovereign_brain_logic(sequence, frequency):
    logic = ''
    for nucleotide, count in frequency.items():
        if count > 1:
            logic += f'{nucleotide} {count} times\n'
        else:
            logic += f'{nucleotide} once\n'
    return logic

neon_sequence = generate_neon_sequence(100)
frequency = neon_analysis(neon_sequence)
optimized_logic = optimize_sovereign_brain_logic(neon_sequence, frequency)

print(optimized_logic)