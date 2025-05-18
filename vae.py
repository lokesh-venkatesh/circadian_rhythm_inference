# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from model import *
from utils import *
from config import *

data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

fourier = lambda x: np.stack([np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)

starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24)) % 365
seasonal_data = fourier(data_days/365)

training_ratio = 0.8

train = data.values[:int(len(data)*training_ratio)]
test = data.values[int(len(data)*training_ratio):]
train_seasonal = seasonal_data[:int(len(data)*training_ratio)]
test_seasonal = seasonal_data[int(len(data)*training_ratio):]

# convert to tensors
train_tensor = tf.convert_to_tensor(train, dtype=tf.float32)
test_tensor = tf.convert_to_tensor(test, dtype=tf.float32)
train_seasonal_tensor = tf.convert_to_tensor(train_seasonal, dtype=tf.float32)
test_seasonal_tensor = tf.convert_to_tensor(test_seasonal, dtype=tf.float32)

vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior)
optimizer = Adam(learning_rate=learning_rate)
vae.compile(optimizer=optimizer)

history = vae.fit(
    train_tensor, train_seasonal_tensor, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_tensor, test_seasonal_tensor),
)

# Model reconstruction accuracy
print('noise emission sigma: ', np.exp(0.5*vae.noise_log_var)[0])

# Loss on validation data
recon_loss, kl_loss = vae.vae_loss((test_tensor, test_seasonal_tensor))
recon_loss = recon_loss.numpy()
kl_loss = kl_loss.numpy()
total_loss = recon_loss + kl_loss

print('Reconstruction loss: ', recon_loss)
print('KL loss:             ', kl_loss)
print('Total loss:          ', total_loss)

encoded_mean, encoded_log_var, encoded_z = encoder(train_tensor)

# Save latent vectors
latent_vectors = encoded_mean.numpy()  # Shape: (num_samples, latent_dim, latent_filter)
flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)  # Flatten for saving

# Save as CSV
latent_df = pd.DataFrame(flat_latents)
latent_df.index = data.index[:len(latent_df)]  # Optional: align with input time series
latent_df.to_csv("data/latent_vectors.csv")

# Save as .npy for quick load
np.save("data/latent_vectors.npy", latent_vectors)

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
end_date = '2020-12-31 17:00:00'
dt = pd.date_range(start=start_date, end=end_date, freq='h')

gen_seasonal_inputs = fourier((dt.dayofyear[::4*24]-1)/365)[np.newaxis]

_, _, z_gen = seasonal_prior(gen_seasonal_inputs)

gen_mean = decoder(z_gen).numpy()
noise = np.random.normal(size=gen_mean.shape)*np.exp(0.5*vae.noise_log_var[0].numpy())
gen = gen_mean #+ noise

gen_series = pd.Series(gen[0, :len(dt)]*18.29 + 75.08, index=dt)
gen_series = gen_series.rename('temperature')
gen_series = gen_series.to_frame()
gen_series.index.name = 'time'

gen_series.to_csv('data/processed/generated.csv')