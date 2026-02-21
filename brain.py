import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Neon DNA sequence data
data = pd.read_csv('neon_dna_sequence.csv')

# Scale the data using StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA on the scaled data
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Perform K-Means clustering on the PCA-transformed data
kmeans = KMeans(n_clusters=3)
data_kmeans = kmeans.fit_transform(data_pca)

# Calculate the silhouette score
silhouette = silhouette_score(data_pca, data_kmeans.labels_)
print(f'Silhouette Score: {silhouette:.3f}')