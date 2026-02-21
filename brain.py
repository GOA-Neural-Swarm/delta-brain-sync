import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the neon DNA sequence data
neon_data = pd.read_csv("neon_dna_sequence.csv")

# Convert the DNA sequence into a numerical representation using a one-hot encoding scheme
one_hot_encoder = np.zeros((neon_data.shape[0], 4))
for i, row in neon_data.iterrows():
    one_hot_encoder[i, :] = [int(row['A']), int(row['C']), int(row['G']), int(row['T'])]

# Scale the one-hot encoded data using StandardScaler
scaler = StandardScaler()
one_hot_scaled = scaler.fit_transform(one_hot_encoder)

# Perform dimensionality reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(one_hot_scaled)

# Visualize the reduced dimensional representation using a scatter plot
import matplotlib.pyplot as plt
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c='b', s=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Neon DNA Sequence Analysis')
plt.show()