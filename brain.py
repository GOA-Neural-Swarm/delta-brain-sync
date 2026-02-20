import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the input data
X = np.array([PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering to group similar data points
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pca)
labels = kmeans.labels_

# Train a logistic regression model on the clustered data
model = LogisticRegression()
model.fit(X_pca, labels)

# Evaluate the model's accuracy
y_pred = model.predict(X_pca)
accuracy = accuracy_score(y, labels)
print(f"Accuracy: {accuracy:.3f}")

# Print the confusion matrix
conf_mat = confusion_matrix(y, labels)
print(conf_mat)