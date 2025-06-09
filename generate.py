# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

from model import build_encoder, build_decoder, build_seasonal_prior, VAE
from utils import set_seed
from config import *


os.makedirs('data/processed', exist_ok=True)
set_seed()
data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

'''
fourier = lambda x: np.stack(
    [np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + 
    [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)
'''

# To demonstrate loading, create a new instance and load weights
encoder = build_encoder()
decoder = build_decoder()
seasonal_prior = build_seasonal_prior()
# vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior)
vae = VAE(encoder=encoder, decoder=decoder)
# optimizer = Adam(learning_rate=learning_rate)
vae.compile()

'''
starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24))%365
seasonal_data = fourier(data_days/365)
'''

gen_dataset = data.values
'''gen_dataset_seasonal = seasonal_data'''

# convert to tensors
gen_dataset_tensor = tf.convert_to_tensor(gen_dataset, dtype=tf.float32)
'''gen_dataset_seasonal_tensor = tf.convert_to_tensor(gen_dataset_seasonal, dtype=tf.float32)'''

# Call the model once to build it
dummy_input = tf.zeros((1, gen_dataset.shape[1]), dtype=tf.float32)  # shape (1, 1536)
_ = vae(dummy_input)
vae.load_weights('model/model_weights.weights.h5')

encoded_mean, encoded_log_var, encoded_z = encoder(gen_dataset_tensor)

# Save latent vectors
latent_vectors = encoded_mean.numpy()  # Shape: (num_samples, latent_dim, latent_filter)
flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)  # Flatten for saving

# Save as CSV
latent_df = pd.DataFrame(flat_latents)
latent_df.index = data.index[:len(latent_df)]  # Optional: align with input time series
latent_df.to_csv("data/processed/latent_vectors.csv")

# Save as .npy for quick load
np.save("data/processed/latent_vectors.npy", latent_vectors)

# set figure size
plt.figure(figsize=(15, 5))
# boxplot of encoded_mean
plt.subplot(1, 2, 1)
plt.boxplot(encoded_mean.numpy().reshape(-1, latent_filter))
plt.title('Encoded Mean')

# boxplot of encoded_log_var
plt.subplot(1, 2, 2)
plt.boxplot(encoded_log_var.numpy().reshape(-1, latent_filter))
plt.title('Encoded Log Variance')

plt.savefig("images/encoded_log_variance.png", dpi=300)

start_date = '1970-01-01 00:00:00'
end_date = '2020-12-31 16:00:00' 
dt = pd.date_range(start=start_date, end=end_date, freq='h')

'''gen_seasonal_inputs = fourier((dt.dayofyear[::LATENT_SIZE])/365)[np.newaxis]

_, _, z_gen = seasonal_prior(gen_seasonal_inputs)

gen_mean = decoder(z_gen).numpy()
noise = np.random.normal(size=gen_mean.shape)*np.exp(0.5*vae.noise_log_var[0].numpy())
gen = gen_mean #+ noise
'''

#n_latents = len(dt) // LATENT_SIZE
n_latents = int(np.ceil(len(dt) / LATENT_SIZE))
z_gen = np.random.normal(size=(1, n_latents, latent_filter)).astype(np.float32)
gen_mean = decoder(z_gen).numpy()

# Add noise (optional, commented out)
noise = np.random.normal(size=gen_mean.shape) * np.exp(0.5 * vae.noise_log_var[0].numpy())
gen = gen_mean + noise
gen = gen_mean  # without noise

data_params = pd.read_csv('data/data_params.csv', index_col=0)
offset = data_params.loc['offset', 'values']
scale = data_params.loc['scale', 'values']

gen_series = pd.Series(gen[0, :len(dt)]*scale + offset, index=dt)
gen_series = gen_series.rename('temperature')
gen_series = gen_series.to_frame()
gen_series.index.name = 'time'

gen_series.to_csv('data/processed/generated.csv')

print(f"Succesfully generated time series of length: {len(gen_series.iloc[:,0])}")