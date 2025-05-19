import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import os

import matplotlib.pyplot as plt
import plotly.express as px

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

# Compute day_of_year as a fraction (0-1) for color-coding
day_of_year_frac = data.index.dayofyear / 365.0

flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1) # Flatten latent vectors

# ---------------------------------------
# t-SNE - 3D
# ---------------------------------------
print("Running t-SNE (3D)... This may take a while.")
tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
latent_3d_tsne = tsne_3d.fit_transform(flat_latents)

# Static matplotlib t-SNE plot (3D)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(latent_3d_tsne[:, 0], latent_3d_tsne[:, 1], latent_3d_tsne[:, 2],
                c=day_of_year_frac, cmap='viridis', s=12, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Day of Year (fraction)')
ax.set_title("t-SNE - Latent Space (3D)", fontsize=14)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
plt.tight_layout()
plt.savefig("images/latent_space_tsne_3d_DOY.png", dpi=300)
plt.close()

# Interactive plotly t-SNE (3D)
fig_tsne_plotly = px.scatter_3d(
    x=latent_3d_tsne[:, 0], y=latent_3d_tsne[:, 1], z=latent_3d_tsne[:, 2],
    color=day_of_year_frac,
    title="t-SNE - Latent Space (3D, Interactive)",
    labels={"x": "t-SNE 1", "y": "t-SNE 2", "z": "t-SNE 3", "color": "Day of Year (fraction)"},
    opacity=0.7,
    color_continuous_scale='Viridis'
)
fig_tsne_plotly.write_html("images/latent_space_tsne_3d_interactive_DOY.html")

# ---------------------------------------
# t-SNE - 2D Pairwise Plots
# ---------------------------------------
pair_labels = [("t-SNE 1", "t-SNE 2"), ("t-SNE 1", "t-SNE 3"), ("t-SNE 2", "t-SNE 3")]
pairs = [(0, 1), (0, 2), (1, 2)]

for (i, j), (xlabel, ylabel) in zip(pairs, pair_labels):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(latent_3d_tsne[:, i], latent_3d_tsne[:, j],
                     c=day_of_year_frac, cmap='viridis', s=12, alpha=0.8)
    plt.colorbar(sc, label='Day of Year (fraction)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"t-SNE - Latent Space ({xlabel} vs {ylabel})")
    plt.tight_layout()
    plt.savefig(f"images/latent_space_tsne_{i+1}_{j+1}_DOY.png", dpi=300)
    plt.close()