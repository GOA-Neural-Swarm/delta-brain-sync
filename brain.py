import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Neon DNA Sequence data
neon_data = pd.read_csv('neon_dna_sequence.csv')

# Preprocess the data
scaler = StandardScaler()
neon_data[['sequence']] = scaler.fit_transform(neon_data[['sequence']])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
neon_data[['sequence_pca']] = pca.fit_transform(neon_data[['sequence']])

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
neon_data[['sequence_tsne']] = tsne.fit_transform(neon_data[['sequence_pca']])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
neon_data[['cluster']] = kmeans.fit_predict(neon_data[['sequence_tsne']])

# Calculate silhouette scores
silhouette_scores = silhouette_score(neon_data[['sequence_tsne']], neon_data[['cluster']])
print("Silhouette score:", silhouette_scores)

# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(neon_data[['sequence_tsne']].iloc[:, 0], neon_data[['sequence_tsne']].iloc[:, 1], c=neon_data[['cluster']])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('K-Means Clustering on Neon DNA Sequence')
plt.show()