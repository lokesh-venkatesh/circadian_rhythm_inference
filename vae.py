# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from utils import *

# define constants
INPUT_SIZE = 64*24 # dimensions of the input data
DEGREE = 3 # degree of fourier series for seasonal inputs
LATENT_SIZE = 4*24 # the hours associated with each latent variable

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




# Parameters
input_shape = None #INPUT_SIZE
latent_dim = None #INPUT_SIZE//LATENT_SIZE
latent_filter = 10
interim_filters = 2*latent_filter

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

encoder = build_encoder()

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

decoder = build_decoder()

def build_seasonal_prior():
    seasonal_inputs = layers.Input(shape=(latent_dim, 2*DEGREE,))
    x = layers.Dense(2*latent_filter, use_bias=False)(seasonal_inputs)
    z_mean = x[:, :, :latent_filter]
    z_log_var = x[:, :, latent_filter:]
    z = Sampling()([z_mean, z_log_var])
    seasonal_prior = models.Model(seasonal_inputs, [z_mean, z_log_var, z], name='seasonal_prior')
    seasonal_prior.summary()
    return seasonal_prior

seasonal_prior = build_seasonal_prior()

class VAE(models.Model):
    def __init__(self, encoder, decoder, prior, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.noise_log_var = self.add_weight(name='var', shape=(1,), initializer='zeros', trainable=True)

    @tf.function
    def vae_loss(self, data):
        values, seasonal = data
        z_mean, z_log_var, z = self.encoder(values)
        reconstructed = self.decoder(z)
        reconstruction_loss = -log_lik_normal_sum(values, reconstructed, self.noise_log_var)/INPUT_SIZE
        seasonal_z_mean, seasonal_z_log_var, _ = self.prior(seasonal)
        kl_loss_z = kl_divergence_sum(z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var)/INPUT_SIZE
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

vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior)






learning_rate = 0.001

optimizer = Adam(learning_rate=learning_rate)

vae.compile(optimizer=optimizer)

epochs = 500

batch_size = 32

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