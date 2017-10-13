# Fitting a VAE on the Kaggle - planet data
# Stijn Decubber

import os
import sys
import time
import shutil

import argparse
import json
import pandas as pd
import numpy as np
import keras as k
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

# Import modules
import features.feature_utils as fu
import models.model_utils as mu
import models.models as m
import models.F_optimizers as fo
import plots.plot_utils as pu
import log_utils as lu
import dir_utils as du

# Import extended datagenerator
# Source: https://gist.github.com/jandremarais/6bf673c76203f612f5ab2981430eb2ef
# See also: https://github.com/fchollet/keras/issues/3296
# !!! The order in which the generator yields the files is not the same order as in the folder structure!
# Though it is the same ordering as listed by os.listdirt(directory)
# This has to do with the way the filesystem orders the files
# Solution: make sure to map the predictions correctly

import extended_generator

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

def train_VAE(argparser):
    args = argparser.parse_args()
    start_time = time.time()

    train_dir = '../data/raw/train-jpg' # Use original training data for validation
    test_dir = '../data/raw/test-jpg' # Use test data for training! (n_test >> n_train images)

    # Data generator
    gen_no_augmentation = extended_generator.ImageDataGenerator(rescale=1./255)

    training_generator = gen_no_augmentation.flow_from_directory(test_dir, target_size=(args.size,args.size),
         class_mode=None, batch_size=args.batch_size)
    validation_generator = gen_no_augmentation.flow_from_directory(train_dir, target_size=(args.size, args.size),
        class_mode=None, batch_size=args.batch_size)

    # 1. Encoder network
    intermediate_dim = 100
    latent_dim = 100

    x = Input(batch_shape=(args.batch_size, args.size))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    # 2. Sampler
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(args.batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # 3. Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    print('Start training...')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train, x_train,
        shuffle=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, x_test))


def main():

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    parser = argparse.ArgumentParser(description='Variational autoencoder on Kaggle planet images')
    parser.add_argument('-e', '--epochs', type=int, help="number of epochs",
                        default=10)
    parser.add_argument('-s', '--size', type=int, choices=(48,64,96,128,256),
                        help='image size used for training', default=48)
    parser.add_argument('-b','--batch_size', type=int, default=32,
                        help='batch size')
    train_VAE(parser)

if __name__ == "__main__":
    sys.exit(main())
