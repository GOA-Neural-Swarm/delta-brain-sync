import random
import string

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def brain_upgrade(code):
    upgraded_code = ""
    for char in code:
        if char in string.ascii_letters:
            upgraded_code += chr(ord(char) + 1)
        elif char in string.digits:
            upgraded_code += str(int(char) + 1)
        else:
            upgraded_code += char
    return upgraded_code

def neural_network(input_string):
    output_string = ""
    for char in input_string:
        if char in string.ascii_letters:
            output_string += chr(ord(char) - 1)
        elif char in string.digits:
            output_string += str(int(char) - 1)
        else:
            output_string += char
    return output_string

def meta_cognition(input_string):
    output_string = neural_network(input_string)
    upgraded_code = brain_upgrade(output_string)
    return upgraded_code

# Prompt definition
input_string = generate_random_string(100)

# Meta-Cognition logic
output_code = meta_cognition(input_string)

print(output_code)