import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Define brain parameters
brain_params = {
    'n_hidden_layers': 2,
    'hidden_layer_sizes': (100, 50),
    'activation_func': 'tanh',
   'solver': 'adam',
   'max_iter': 1000
}

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_redundant=2, n_repeated=0, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize neural network
mlp = MLPClassifier(**brain_params)

# Train neural network
mlp.fit(X_train, y_train)

# Evaluate neural network performance
y_pred = mlp.predict(X_test)
print("Accuracy:", mlp.score(X_test, y_test))