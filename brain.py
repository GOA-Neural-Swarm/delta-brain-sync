import random
import string
import time

# Function to generate a random DNA sequence
def generate_dna_sequence(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# Function to calculate the similarity between two DNA sequences
def calculate_similarity(seq1, seq2):
    similarity = 0
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            similarity += 1
    return similarity / len(seq1)

# Function to evolve the DNA sequence
def evolve_dna_sequence(seq, iterations):
    for _ in range(iterations):
        new_seq = generate_dna_sequence(len(seq))
        if calculate_similarity(seq, new_seq) > calculate_similarity(seq, seq):
            seq = new_seq
    return seq

# Main function
def main():
    # Initialize the DNA sequence
    dna_sequence = generate_dna_sequence(100)
    print("Initial DNA sequence:", dna_sequence)

    # Evolve the DNA sequence
    evolved_dna_sequence = evolve_dna_sequence(dna_sequence, 1000)
    print("Evolved DNA sequence:", evolved_dna_sequence)

    # Print the final DNA sequence
    print("Final DNA sequence:", evolved_dna_sequence)

# Run the main function
if __name__ == "__main__":
    main()