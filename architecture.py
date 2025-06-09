# visualize_model.py

import tensorflow as tf
from tensorflow.keras.utils.vis_utils import plot_model
from model import build_encoder, build_decoder, build_seasonal_prior, VAE
from config import input_shape  # import other constants like latent_dim, latent_filter, interim_filters, DEGREE if needed
import os

# Create output directory
os.makedirs("model", exist_ok=True)

# 1. Build components
encoder = build_encoder()
decoder = build_decoder()
seasonal_prior = build_seasonal_prior()

# 2. Save individual models for Netron
encoder.save("model/encoder_model.h5")
decoder.save("model/decoder_model.h5")
seasonal_prior.save("model/seasonal_prior_model.h5")

# 3. Reconstruct full VAE and call it once with dummy input
vae = VAE(encoder, decoder, seasonal_prior)
dummy_input = tf.zeros((1, input_shape))
vae(dummy_input)  # This builds the model

# 4. Save the full VAE model
vae.save("model/full_vae.keras")

# 5. Optional: Plot encoder and decoder architecture
plot_model(encoder, to_file="model/encoder_architecture.png", show_shapes=True, expand_nested=True)
plot_model(decoder, to_file="model/decoder_architecture.png", show_shapes=True, expand_nested=True)

print("✅ Saved encoder, decoder, and full VAE models to 'model/'.")
print("🧠 You can now open them in https://netron.app to visualize the architectures.")
