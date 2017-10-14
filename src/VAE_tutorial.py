# VAE tutorial with Keras
# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler


m = 50
n_z = 10 # number of latent factores
n_epoch = 20

# Define the components

# Q(z|X) -- encoder
inputs = Input(shape=(784,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

# Reparametization trick: move the sampling outside of the network as an additional input
# https://www.quora.com/What-is-the-reparameterization-trick-in-variational-autoencoders
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps

# Sample z ~ Q(z|X)
# Lambda wraps arbitrary transformation as a Layer object
z = Lambda(sample_z)([mu, log_sigma])

# P(X|z) -- decoder
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# Put components together

# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train = mnist.train.images[:,:]

print('Training the VAE...')

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(X_train, X_train, batch_size=m, epochs=n_epoch, verbose=1)

print('Done!')

# Generate some samples

fig, axes = plt.subplots(nrows=5, ncols=5)
for axrow in axes:
    for ax in axrow:
        sample_z = np.random.normal(size=(n_z),loc=0., scale=1.).reshape(-1,n_z)
        sample_x = decoder.predict(sample_z)
        ax.imshow(sample_x.reshape((28,28)))

plt.show()
fig.savefig('./vae.png')
