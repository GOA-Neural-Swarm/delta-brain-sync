def sigmoid_derivative(x):
    return x * (1 - x)

def dReLU(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x
