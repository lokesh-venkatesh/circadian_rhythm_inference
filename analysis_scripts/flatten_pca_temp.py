import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import os

os.makedirs('images', exist_ok=True)

# Load data
latent_vectors = np.load("data/processed/latent_vectors.npy")
data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

print(f"Length of the latent vectors dataset: {latent_vectors.shape[0]}")
print(f"Length of the time-series vectors dataset: {data.shape[0]}")

# Align number of latent vectors and temperature data
if latent_vectors.shape[0] != data.shape[0]:
    print(f"Mismatch: {latent_vectors.shape[0]} latent vectors vs {data.shape[0]} rows in CSV. Trimming data.")
    data = data.iloc[:latent_vectors.shape[0]]

# Label for color-coding
mean_temps = data.mean(axis=1)
flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1) # Flatten latent vectors

# ---------------------------------------
# PCA - 3D
# ---------------------------------------
pca = PCA(n_components=3)
latent_3d_pca = pca.fit_transform(flat_latents)

# Static matplotlib PCA plot (3D)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(latent_3d_pca[:, 0], latent_3d_pca[:, 1], latent_3d_pca[:, 2],
                c=mean_temps, cmap='viridis', s=12, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Mean Temperature')
ax.set_title("PCA - Latent Space (3D)", fontsize=14)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.tight_layout()
plt.savefig("images/latent_space_pca_3d_TEMP.png", dpi=300)
plt.close()

# Interactive plotly PCA (3D)
fig_pca_plotly = px.scatter_3d(
    x=latent_3d_pca[:, 0], y=latent_3d_pca[:, 1], z=latent_3d_pca[:, 2],
    color=mean_temps,
    title="PCA - Latent Space (3D, Interactive)",
    labels={"x": "PC 1", "y": "PC 2", "z": "PC 3", "color": "Mean Temp"},
    opacity=0.7,
    color_continuous_scale='Viridis'
)
fig_pca_plotly.write_html("images/latent_space_pca_3d_interactive_TEMP.html")

# ---------------------------------------
# PCA - 2D Pairwise Plots
# ---------------------------------------
pair_labels = [("PC 1", "PC 2"), ("PC 1", "PC 3"), ("PC 2", "PC 3")]
pairs = [(0, 1), (0, 2), (1, 2)]

for (i, j), (xlabel, ylabel) in zip(pairs, pair_labels):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(latent_3d_pca[:, i], latent_3d_pca[:, j],
                     c=mean_temps, cmap='viridis', s=12, alpha=0.8)
    plt.colorbar(sc, label='Mean Temperature')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"PCA - Latent Space ({xlabel} vs {ylabel})")
    plt.tight_layout()
    plt.savefig(f"images/latent_space_pca_{i+1}_{j+1}_TEMP.png", dpi=300)
    plt.close()