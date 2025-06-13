import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os

# Setup
os.makedirs('images', exist_ok=True)

filename_without_extension = "training_run_Seasonal_prior_True_phase_shift_100_epochs_latent_vectors" #"data/processed/latent_vectors"
to_save_plot_to = "training_run_Seasonal_prior_True_phase_shift_100_epochs_latent_space_analysis.png" # images/latent_space_analysis.png

# Load data
latent_vectors = np.load(f"{filename_without_extension}.npy")
data = pd.read_csv(f'{filename_without_extension}.csv', index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index, format='mixed')

if latent_vectors.shape[0] != data.shape[0]:
    print(f"Mismatch: {latent_vectors.shape[0]} latent vectors vs {data.shape[0]} rows in CSV. Trimming data.")
    data = data.iloc[:latent_vectors.shape[0]]

# Colour labels
mean_temps = data.mean(axis=1)
day_of_year_frac = data.index.dayofyear / 365.0
hour_of_day_frac = data.index.hour / 24.0

# Flatten latent vectors
flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)

# Reduce dimensions
print("First performing a PCA")
pca = PCA(n_components=2)
pca_latents = pca.fit_transform(flat_latents)
print("PCA done")

print("Now performing t-SNE")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_latents = tsne.fit_transform(flat_latents)
print("t-SNE done")

print("Now performing UMAP")
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_latents = umap_model.fit_transform(flat_latents)
print("UMAP done")

# Helper for plotting
colorings = {
    "Mean Temp": mean_temps,
    "Day of Year (Fraction)": day_of_year_frac,
    "Hour of Day (Fraction)": hour_of_day_frac
}

reduced_spaces = {
    "Raw": flat_latents,
    "PCA": pca_latents,
    "t-SNE": tsne_latents,
    "UMAP": umap_latents
}

# Plot
fig, axs = plt.subplots(4, 3, figsize=(18, 20))

# Define colormaps for each coloring
cmap_dict = {
    "Mean Temp": "coolwarm",
    "Day of Year (Fraction)": "viridis",
    "Hour of Day (Fraction)": "viridis"
}

for i, (method_name, embedding) in enumerate(reduced_spaces.items()):
    for j, (label_name, label_values) in enumerate(colorings.items()):
        print(f"Plotting {method_name} coloured by {label_name}")
        ax = axs[i, j]
        cmap = cmap_dict.get(label_name, "viridis")
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=label_values, cmap=cmap, s=8, alpha=0.7)
        ax.set_title(f"{method_name} - Coloured by {label_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.01)

plt.tight_layout()
plt.savefig(to_save_plot_to, dpi=500)
plt.close()
print(f"Saved mega plot to {to_save_plot_to}")
