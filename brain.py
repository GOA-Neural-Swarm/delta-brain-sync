import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
X = np.load('data.npy')
y = np.load('labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

# Train the model using the training data
model.fit(X_train_scaled, y_train)

# Evaluate the model using the testing data
y_pred = model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Evolve the neural network using evolutionary algorithms
def evolve_model(population_size, generations):
    population = [model.deepcopy() for _ in range(population_size)]
    for _ in range(generations):
        fitness_scores = []
        for model in population:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            fitness_scores.append(np.mean(y_pred == y_test))
        parents = np.array(population)[np.argsort(fitness_scores)][-population_size//2:]
        offspring = []
        for _ in range(population_size//2):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child = model.deepcopy()
            child.hidden_layer_sizes = [(100 + parent1.hidden_layer_sizes[0] + parent2.hidden_layer_sizes[0]) // 2,
                                        (50 + parent1.hidden_layer_sizes[1] + parent2.hidden_layer_sizes[1]) // 2]
            child.max_iter = (parent1.max_iter + parent2.max_iter) // 2
            offspring.append(child)
        population = offspring
    return population[0]

evolved_model = evolve_model(population_size=100, generations=100)

# Evaluate the evolved model
y_pred = evolved_model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)
print(f'Evolved model accuracy: {accuracy:.4f}')