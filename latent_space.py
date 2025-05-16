import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Load data
latent_vectors = np.load("data/latent_vectors.npy")
data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

'''REMOVE THE BELOW CODE LATER???'''
# Align number of latent vectors and temperature data
if latent_vectors.shape[0] != data.shape[0]:
    print(f"Mismatch: {latent_vectors.shape[0]} latent vectors vs {data.shape[0]} rows in CSV. Trimming data.")
    data = data.iloc[:latent_vectors.shape[0]]

# Label for color-coding
mean_temps = data.mean(axis=1)
'''REMOVE THE ABOVE CODE LATER???'''

mean_temps = data.mean(axis=1)  # Label for color-coding

# Flatten latent vectors
flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)

# ---------------------------------------
# PCA - 3D
# ---------------------------------------
pca = PCA(n_components=3)
latent_3d_pca = pca.fit_transform(flat_latents)

# Static matplotlib PCA plot
sns.set(style="whitegrid")
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
plt.savefig("images/latent_space_pca_3d.png", dpi=300)
plt.close()

# Interactive plotly PCA
fig_pca_plotly = px.scatter_3d(
    x=latent_3d_pca[:, 0], y=latent_3d_pca[:, 1], z=latent_3d_pca[:, 2],
    color=mean_temps,
    title="PCA - Latent Space (3D, Interactive)",
    labels={"x": "PC 1", "y": "PC 2", "z": "PC 3", "color": "Mean Temp"},
    opacity=0.7,
    color_continuous_scale='Viridis'
)
fig_pca_plotly.write_html("images/latent_space_pca_3d_interactive.html")

# ---------------------------------------
# t-SNE - 2D
# ---------------------------------------
tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
latent_2d_tsne = tsne_2d.fit_transform(flat_latents)

plt.figure(figsize=(10, 8))
sc = plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                 c=mean_temps, cmap='plasma', s=12, alpha=0.8)
plt.colorbar(sc, label='Mean Temperature')
plt.title("t-SNE - Latent Space (2D)", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("images/latent_space_tsne_2d.png", dpi=300)
plt.close()

# ---------------------------------------
# t-SNE - 3D
# ---------------------------------------
tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=42)
latent_3d_tsne = tsne_3d.fit_transform(flat_latents)

# Static matplotlib t-SNE 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(latent_3d_tsne[:, 0], latent_3d_tsne[:, 1], latent_3d_tsne[:, 2],
                c=mean_temps, cmap='plasma', s=12, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Mean Temperature')
ax.set_title("t-SNE - Latent Space (3D)", fontsize=14)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
plt.tight_layout()
plt.savefig("images/latent_space_tsne_3d.png", dpi=300)
plt.close()

# Interactive plotly t-SNE 3D
fig_tsne_plotly = px.scatter_3d(
    x=latent_3d_tsne[:, 0], y=latent_3d_tsne[:, 1], z=latent_3d_tsne[:, 2],
    color=mean_temps,
    title="t-SNE - Latent Space (3D, Interactive)",
    labels={"x": "t-SNE 1", "y": "t-SNE 2", "z": "t-SNE 3", "color": "Mean Temp"},
    opacity=0.7,
    color_continuous_scale='Plasma'
)
fig_tsne_plotly.write_html("images/latent_space_tsne_3d_interactive.html")
