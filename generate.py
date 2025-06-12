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

if prior_dist_type == 'Normal':
    encoder = build_encoder()
    decoder = build_decoder()
    seasonal_prior = build_seasonal_prior()
    vae = VAE(encoder=encoder, decoder=decoder, prior_dist_type=prior_dist_type)
    vae.compile()

    gen_dataset = data.values
    gen_dataset_tensor = tf.convert_to_tensor(gen_dataset, dtype=tf.float32)

    dummy_input = tf.zeros((1, gen_dataset.shape[1]), dtype=tf.float32)
    _ = vae(dummy_input)
    vae.load_weights('model/model_weights.weights.h5')

    encoded_mean, encoded_log_var, encoded_z = encoder(gen_dataset_tensor)

    # Save latent vectors
    latent_vectors = encoded_mean.numpy()
    flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)

    latent_df = pd.DataFrame(flat_latents)
    latent_df.index = data.index[:len(latent_df)]
    latent_df.to_csv("data/processed/latent_vectors.csv")
    np.save("data/processed/latent_vectors.npy", latent_vectors)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(encoded_mean.numpy().reshape(-1, latent_filter))
    plt.title('Encoded Mean')

    plt.subplot(1, 2, 2)
    plt.boxplot(encoded_log_var.numpy().reshape(-1, latent_filter))
    plt.title('Encoded Log Variance')

    plt.savefig("images/encoded_log_variance.png", dpi=300)

    dt = pd.date_range(start=gnrt_start, end=gnrt_end, freq='h')

    n_latents = int(np.ceil(len(dt) / LATENT_SIZE))
    z_gen = np.random.normal(size=(1, n_latents, latent_filter)).astype(np.float32)
    gen_mean = decoder(z_gen).numpy()

    # Add noise (optional, commented out)
    noise = np.random.normal(size=gen_mean.shape) * np.exp(0.5 * vae.noise_log_var[0].numpy())
    gen = gen_mean  # without noise

    data_params = pd.read_csv('data/data_params.csv', index_col=0)
    offset = data_params.loc['offset', 'values']
    scale = data_params.loc['scale', 'values']

    gen_series = pd.Series(gen[0, :len(dt)]*scale + offset, index=dt)
    gen_series = gen_series.rename('temperature')
    gen_series = gen_series.to_frame()
    gen_series.index.name = 'time'

    # Add back seasonal cycle
    df_obs = pd.read_csv('data/processed/observed_time_series.csv', index_col=0, parse_dates=True)
    df_obs['weekofyear'] = df_obs.index.isocalendar().week
    weekly_climatology = df_obs.groupby('weekofyear')['Observed'].mean()

    week_numbers = gen_series.index.isocalendar().week
    seasonal_cycle = np.array([weekly_climatology[w] for w in week_numbers])
    gen_series['temperature'] += seasonal_cycle

    gen_series.to_csv('data/processed/generated_time_series.csv')

    print(f"Succesfully generated time series of length: {len(gen_series.iloc[:,0])}")

elif prior_dist_type == 'Seasonal':
    # Fourier features
    fourier = lambda x: np.stack(
        [np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + 
        [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)

    encoder = build_encoder()
    decoder = build_decoder()
    seasonal_prior = build_seasonal_prior()
    vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior, prior_dist_type=prior_dist_type)
    vae.compile()

    starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
    data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24))%365
    seasonal_data = fourier(data_days/365)

    gen_dataset = data.values
    gen_dataset_seasonal = seasonal_data

    gen_dataset_tensor = tf.convert_to_tensor(gen_dataset, dtype=tf.float32)
    gen_dataset_seasonal_tensor = tf.convert_to_tensor(gen_dataset_seasonal, dtype=tf.float32)

    dummy_input = tf.zeros((1, gen_dataset.shape[1]), dtype=tf.float32)
    _ = vae(dummy_input)
    vae.load_weights('model/model_weights.weights.h5')

    encoded_mean, encoded_log_var, encoded_z = encoder(gen_dataset_tensor)

    # Save latent vectors
    latent_vectors = encoded_mean.numpy()
    flat_latents = latent_vectors.reshape(latent_vectors.shape[0], -1)

    latent_df = pd.DataFrame(flat_latents)
    latent_df.index = data.index[:len(latent_df)]
    latent_df.to_csv("data/processed/latent_vectors.csv")
    np.save("data/processed/latent_vectors.npy", latent_vectors)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(encoded_mean.numpy().reshape(-1, latent_filter))
    plt.title('Encoded Mean')

    plt.subplot(1, 2, 2)
    plt.boxplot(encoded_log_var.numpy().reshape(-1, latent_filter))
    plt.title('Encoded Log Variance')

    plt.savefig("images/encoded_log_variance.png", dpi=300)

    dt = pd.date_range(start=gnrt_start, end=gnrt_end, freq='h')

    gen_seasonal_inputs = fourier((dt.dayofyear[::LATENT_SIZE])/365)[np.newaxis]
    _, _, z_gen = seasonal_prior(gen_seasonal_inputs)
    gen_mean = decoder(z_gen).numpy()
    noise = np.random.normal(size=gen_mean.shape)*np.exp(0.5*vae.noise_log_var[0].numpy())
    gen = gen_mean  # without noise

    data_params = pd.read_csv('data/data_params.csv', index_col=0)
    offset = data_params.loc['offset', 'values']
    scale = data_params.loc['scale', 'values']

    gen_series = pd.Series(gen[0, :len(dt)]*scale + offset, index=dt)
    gen_series = gen_series.rename('temperature')
    gen_series = gen_series.to_frame()
    gen_series.index.name = 'time'

    # Add back seasonal cycle
    df_obs = pd.read_csv('data/processed/observed_time_series.csv', index_col=0, parse_dates=True)
    df_obs['weekofyear'] = df_obs.index.isocalendar().week
    weekly_climatology = df_obs.groupby('weekofyear')['Observed'].mean()

    week_numbers = gen_series.index.isocalendar().week
    seasonal_cycle = np.array([weekly_climatology[w] for w in week_numbers])
    gen_series['temperature'] += seasonal_cycle

    gen_series.to_csv('data/processed/generated.csv')

    print(f"Succesfully generated time series of length: {len(gen_series.iloc[:,0])}")

    gen_series['temperature'].resample('M').mean().plot(title='Monthly Mean of Generated Series')
    plt.savefig('images/generated_monthly_avg.png', dpi=300)