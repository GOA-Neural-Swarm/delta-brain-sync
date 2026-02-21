import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load DNA sequence data
dna_data = pd.read_csv('neon_dna_sequence_analysis.csv')

# Preprocess data (e.g., normalize, scale)
scaler = StandardScaler()
dna_data[['sequence']] = scaler.fit_transform(dna_data[['sequence']])

# Perform PCA dimensionality reduction
pca = PCA(n_components=2)
dna_data_pca = pca.fit_transform(dna_data[['sequence']])

# Cluster data using K-Means
kmeans = KMeans(n_clusters=3)
dna_data_clusters = kmeans.fit_predict(dna_data_pca)

# Calculate silhouette score for cluster evaluation
silhouette_values = silhouette_score(dna_data_pca, dna_data_clusters)

print("Silhouette score:", silhouette_values)
plt.scatter(dna_data_pca[:, 0], dna_data_pca[:, 1], c=dna_data_clusters)
plt.title("K-Means Clustering of Neon DNA Sequence Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()