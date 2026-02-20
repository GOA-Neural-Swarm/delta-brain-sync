import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

def optimize_brain(dna_sequence):
    # Extract relevant features from DNA sequence
    features = pd.Series([int(dna_sequence[i]) for i in range(len(dna_sequence))])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.values.reshape(-1, 1))
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(features_pca)
    
    # Plot results
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Optimized Brain Clustering')
    plt.show()
    
    return labels

# Run optimization
labels = optimize_brain(Sequence)
print(labels)