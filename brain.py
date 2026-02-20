import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Define the objective function
def objective(params):
    # Unpack parameters
    w1, b1, w2, b2 = params
    
    # Calculate the output
    output = np.tanh(np.dot(X_train, w1) + b1)
    output = np.dot(output, w2) + b2
    
    # Calculate the loss
    loss = np.mean((output - y_train) ** 2)
    
    return loss

# Load the data
np.random.seed(0)
X_train, y_train = np.random.rand(100, 100), np.random.rand(100)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the parameters
params = np.random.rand(4)

# Optimize the parameters
res = minimize(objective, params, method="SLSQP")

# Print the optimized parameters
print(res.x)

# Plot the optimized output
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Optimized Output")
plt.show()