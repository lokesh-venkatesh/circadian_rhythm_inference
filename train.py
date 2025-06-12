# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

from model import *
from utils import *
from config import *

set_seed()
os.makedirs('model', exist_ok=True)
data = pd.read_csv('data/processed/phoenix_64days.csv', index_col=0, parse_dates=True)

if prior_dist_type=='Seasonal':
    fourier = lambda x: np.stack(
        [np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + 
        [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)

    starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
    data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24))%365
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

    encoder = build_encoder()
    decoder = build_decoder()
    seasonal_prior = build_seasonal_prior()

    vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior, prior_dist_type='Seasonal')
    optimizer = Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)

    history = vae.fit(
        train_tensor, train_seasonal_tensor, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_tensor, test_seasonal_tensor),
    )


    # Save model weights
    _ = vae(train_tensor[:1])  # Call model once to build it
    vae.save_weights('model/model_weights.weights.h5')

    # Model reconstruction accuracy
    print('noise emission sigma: ', np.exp(0.5*vae.noise_log_var)[0])

    # Loss on validation data
    recon_loss, kl_loss = vae.vae_loss((test_tensor, test_seasonal_tensor))
    recon_loss = recon_loss.numpy()
    kl_loss = kl_loss.numpy()
    total_loss = recon_loss + kl_loss

    # Save loss information as a transposed dataframe
    summary_df = pd.DataFrame({
        'Reconstruction loss': [recon_loss],
        'KL loss': [kl_loss],
        'Total loss': [total_loss],
        'Number of epochs': [epochs]
    }).T
    summary_df.columns = ['Training Run Summary']
    summary_df.to_csv('data/train_summary.csv', index=False)

    print('Reconstruction loss: ', recon_loss)
    print('KL loss:             ', kl_loss)
    print('Total loss:          ', total_loss)


elif prior_dist_type=='Normal':
    training_ratio = 0.8
    train = data.values[:int(len(data)*training_ratio)]
    test = data.values[int(len(data)*training_ratio):]

    # convert to tensors
    train_tensor = tf.convert_to_tensor(train, dtype=tf.float32)
    test_tensor = tf.convert_to_tensor(test, dtype=tf.float32)

    encoder = build_encoder()
    decoder = build_decoder()

    vae = VAE(encoder=encoder, decoder=decoder, prior_dist_type='Normal')
    optimizer = Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)

    history = vae.fit(
        train_tensor, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_tensor, ),
    )


    # Save model weights
    _ = vae(train_tensor[:1])  # Call model once to build it
    vae.save_weights('model/model_weights.weights.h5')

    # Model reconstruction accuracy
    print('noise emission sigma: ', np.exp(0.5*vae.noise_log_var)[0])

    # Loss on validation data
    recon_loss, kl_loss = vae.vae_loss(test_tensor)
    recon_loss = recon_loss.numpy()
    kl_loss = kl_loss.numpy()
    total_loss = recon_loss + kl_loss

    # Save loss information as a dataframe
    summary_df = pd.DataFrame({
        'Reconstruction loss': [recon_loss],
        'KL loss': [kl_loss],
        'Total loss': [total_loss]
    })
    summary_df.to_csv('data/train_summary.csv', index=False)

    print('Reconstruction loss: ', recon_loss)
    print('KL loss:             ', kl_loss)
    print('Total loss:          ', total_loss)