import re
import math

def neon_analysis(sequence):
    # Initialize variables
    gc_content = 0
    a_t_count = 0
    c_g_count = 0

    # Count GC content
    for base in sequence:
        if base in ['G', 'C']:
            gc_content += 1
        elif base in ['A', 'T']:
            if base == 'A':
                a_t_count += 1
            else:
                c_g_count += 1

    # Calculate GC content percentage
    gc_percent = (gc_content / len(sequence)) * 100

    # Calculate A-T and C-G counts
    a_t_ratio = a_t_count / len(sequence)
    c_g_ratio = c_g_count / len(sequence)

    # Print results
    print("GC content: {:.2f}%".format(gc_percent))
    print("A-T count: {:.2f}%".format(a_t_ratio * 100))
    print("C-G count: {:.2f}%".format(c_g_ratio * 100))

# Define sequence
sequence = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"

# Perform analysis
neon_analysis(sequence)