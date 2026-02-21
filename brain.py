import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load Neon DNA Sequence Data
neon_data = pd.read_csv('neon_dna_sequence_data.csv')

# Convert categorical variables to numerical variables
neon_data['Species'] = neon_data['Species'].astype('category').cat.codes

# Standardize the data using StandardScaler
scaler = StandardScaler()
neon_data[['Feature1', 'Feature2', 'Feature3']] = scaler.fit_transform(neon_data[['Feature1', 'Feature2', 'Feature3']])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
neon_data[['PC1', 'PC2']] = pca.fit_transform(neon_data[['Feature1', 'Feature2', 'Feature3']])

# Apply TSNE for visualization
tsne = TSNE(n_components=2, random_state=42)
neon_data[['TSNE1', 'TSNE2']] = tsne.fit_transform(neon_data[['PC1', 'PC2']])

# Apply K-Means Clustering for grouping
kmeans = KMeans(n_clusters=4, random_state=42)
neon_data['Cluster'] = kmeans.fit_predict(neon_data[['TSNE1', 'TSNE2']])

# Visualize the results using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=neon_data)
plt.title('Neon DNA Sequence Analysis')
plt.show()