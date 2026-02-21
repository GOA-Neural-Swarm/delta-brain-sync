import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Load the dataset
data = pd.read_csv('neon_dna_sequence_analysis.csv')

# Standardize the data
scaler = StandardScaler()
data[['seq_len', 'GC_content','sequence']] = scaler.fit_transform(data[['seq_len', 'GC_content','sequence']])

# Apply PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data[['seq_len', 'GC_content','sequence']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5)
data_pca['cluster'] = kmeans.fit_predict(data_pca)

# Evaluate the clustering
silhouette = silhouette_score(data_pca, data_pca['cluster'])
calinski_harabasz = calinski_harabasz_score(data_pca, data_pca['cluster'])

print(f"Silhouette score: {silhouette:.4f}")
print(f"Calinski-Harabasz score: {calinski_harabasz:.4f}")

# Visualize the results
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data_pca['cluster'])
plt.title("K-Means Clustering of Neon DNA Sequence Analysis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()