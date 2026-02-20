import re
import random

def reboot_system( sequence, iteration=0 ):
    if iteration > 5:  # Stop recursion after 5 iterations
        return sequence
    randomized_sequence = re.sub(r'([ACGT])', lambda x: random.choice([x.group(1), x.group(1).upper()]), sequence)  # Randomly uppercase or lowercase
    return reboot_system( randomized_sequence, iteration + 1 )

sequence = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"
rebooted_sequence = reboot_system( sequence )
print(rebooted_sequence)