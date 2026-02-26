from numba import njit
import numpy as np

@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit
def sigmoid_derivative(x):
    return x * (1 - x)