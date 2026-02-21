import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load DNA sequence data
dna_data = pd.read_csv("dna_sequence.csv")

# Preprocess data
scaler = StandardScaler()
dna_data[['A', 'C', 'G', 'T']] = scaler.fit_transform(dna_data[['A', 'C', 'G', 'T']])

# Perform PCA
pca = PCA(n_components=2)
dna_data[['PC1', 'PC2']] = pca.fit_transform(dna_data[['A', 'C', 'G', 'T']])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5)
dna_data['cluster'] = kmeans.fit_predict(dna_data[['PC1', 'PC2']])

# Calculate silhouette score
silhouette = silhouette_score(dna_data[['PC1', 'PC2']], dna_data['cluster'])
print("Silhouette score:", silhouette)