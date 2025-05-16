import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

latent_vectors = np.load("data/latent_vectors.npy")
data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)  # Flatten for saving

# Optional: Visualize first 3 principal components
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

pca = PCA(n_components=3)
latent_3d = pca.fit_transform(flat_latents)

# Static matplotlib plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], s=10, alpha=0.7)
ax.set_title("3D PCA of Latent Vectors")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.tight_layout()
plt.savefig("images/latent_space_pca_3d.png", dpi=300)
plt.close()

# Interactive plotly plot
fig_plotly = px.scatter_3d(
    x=latent_3d[:, 0], y=latent_3d[:, 1], z=latent_3d[:, 2],
    title="3D PCA of Latent Vectors",
    labels={"x": "PC 1", "y": "PC 2", "z": "PC 3"},
    opacity=0.7
)
fig_plotly.write_html("images/latent_space_pca_3d_interactive.html")
