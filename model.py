# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import *
from config import *

# Build the encoder
def build_encoder():
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Reshape((-1, 1))(inputs)
    x = layers.Conv1D(interim_filters, 5, strides=3, padding='same', activation='relu')(x)
    x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(2*latent_filter, 3, strides=2, padding='same')(x)
    z_mean = x[: ,:, :latent_filter]
    z_log_var = x[:, :, latent_filter:]
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    return encoder

# Build the decoder
def build_decoder():
    latent_inputs = layers.Input(shape=(latent_dim, latent_filter))
    x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(latent_inputs)
    x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(1, 5, strides=3, padding='same')(x)
    outputs = layers.Reshape((-1,))(x)
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    return decoder

def build_seasonal_prior():
    seasonal_inputs = layers.Input(shape=(None, 2*DEGREE,))
    x = layers.Dense(2*latent_filter, use_bias=False)(seasonal_inputs)
    z_mean = x[:, :, :latent_filter]
    z_log_var = x[:, :, latent_filter:]
    z = Sampling()([z_mean, z_log_var])
    seasonal_prior = models.Model(seasonal_inputs, [z_mean, z_log_var, z], name='seasonal_prior')
    seasonal_prior.summary()
    return seasonal_prior

class VAE(models.Model):
    def __init__(self, encoder, decoder, prior_dist_type, prior=None, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.prior_dist_type = prior_dist_type
        self.noise_log_var = self.add_weight(name='var', shape=(1,), initializer='zeros', trainable=True)

    @tf.function
    def vae_loss(self, data):
        if self.prior_dist_type=='Normal':
            values = data  # Use only values if no seasonal prior is used
            z_mean, z_log_var, z = self.encoder(values)
            reconstructed = self.decoder(z)
            reconstruction_loss = -log_lik_normal_sum(values, reconstructed, self.noise_log_var)/INPUT_SIZE
            kl_loss_z = kl_divergence_standard_normal(z_mean, z_log_var) / INPUT_SIZE 
            return reconstruction_loss, kl_loss_z
        
        elif self.prior_dist_type=='Seasonal':
            values, seasonal = data
            z_mean, z_log_var, z = self.encoder(values)
            reconstructed = self.decoder(z)
            reconstruction_loss = -log_lik_normal_sum(values, reconstructed, self.noise_log_var)/INPUT_SIZE
            seasonal_z_mean, seasonal_z_log_var, _ = self.prior(seasonal)
            kl_loss_z = kl_divergence_sum(z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var)/INPUT_SIZE
            kl_loss_z = kl_divergence_standard_normal(z_mean, z_log_var) / INPUT_SIZE 
            # Standard normal prior: mean=0, log_var=0 (=> std=1)
            return reconstruction_loss, kl_loss_z

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_loss_z = self.vae_loss(data)
            total_loss = reconstruction_loss + kl_loss_z
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': total_loss}
    
    def test_step(self, data):
        reconstruction_loss, kl_loss_z = self.vae_loss(data)

        return {'loss': reconstruction_loss + kl_loss_z, 'recon': reconstruction_loss, 'kl': kl_loss_z}

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
