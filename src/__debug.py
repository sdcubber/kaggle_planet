## Finetune VGGNET


# coding: utf-8
# Planet_flow with VGGnet according to the keras tutorial
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# -------imports---------- #
print('importing stuff...')

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
sys.path.append('../src')

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
# Though it is the same ordering as listed by os.listdir(directory)
# This has to do with the way the filesystem orders the files
# Solution: make sure to map the predictions correctly

import extended_generator
print('Done')

print('Initializing stuff...')
name='test'
epochs=1
size=48
batch_size=32
learning_rate=0.001
threshold=0
iterations=1
TTA=1
optimizer='adam'
debug=False

start_time = time.time()
ts = start_time

temp_training_dir, temp_validation_dir = du.make_temp_dirs(ts, name)
du.empty_validation_folder(temp_training_dir, temp_validation_dir)
du.fill_temp_training_folder(temp_training_dir)
du.move_to_validation_folder(temp_training_dir, temp_validation_dir)

# ------ call data generators ------#
train_directory = os.path.split(temp_training_dir)[0]
validation_directory = os.path.split(temp_validation_dir)[0]
test_directory = '../data/interim/consensus_test/'

# -------load metadata---------- #
labels, df_train, df_test, label_map, train_mapping, validation_mapping, y_train, y_valid = fu.load_metadata(temp_training_dir, temp_validation_dir)

n_train_files = len(os.listdir(os.path.join(train_directory, 'train')))
n_validation_files = len(os.listdir(os.path.join(validation_directory, 'validation')))
n_test_files = len(os.listdir(os.path.join(test_directory, 'test')))

# Generators - with and without data augmentation
gen_no_augmentation = extended_generator.ImageDataGenerator(rescale=1./255)
gen_augmentation = extended_generator.ImageDataGenerator(rotation_range=0, width_shift_range=0.05, height_shift_range=0.05,
    horizontal_flip=True, vertical_flip=True, rescale=1./255)

# Training data: with augmentation - with labels
training_generator = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
class_mode='multilabel', multilabel_classes=train_mapping, n_class=17, batch_size=batch_size)

# Validation data: without augmentation - with labels
validation_generator = gen_no_augmentation.flow_from_directory(validation_directory, target_size=(size,size),
    class_mode='multilabel', multilabel_classes=validation_mapping, n_class=17, batch_size=batch_size, shuffle=False)

print('Done')

# Training data: no augmentation - with labels

print('Saving bottleneck features')

def save_bottlebeck_features(size, datagen,
    training_dir,
    validation_dir,
    train_mapping,
    validation_mapping, name, ts):

    batch_size=100
    # build the VGG16 network
    model = k.applications.VGG16(include_top=False, weights='imagenet')

    # Training data bottleneck features
    generator = datagen.flow_from_directory(training_dir,
        target_size=(size,size),
        class_mode=None,shuffle=False, batch_size=batch_size)

    n_train_files = len(os.listdir(os.path.join(train_directory, 'train')))
    bottleneck_features_train = model.predict_generator(
        generator, n_train_files/batch_size, verbose=1)
    np.save('../models/bottleneck_features_train_{}_{}.npy'.format(ts, name), bottleneck_features_train)

    # Validation data bottleneck features
    generator = datagen.flow_from_directory(validation_dir,
        target_size=(size,size),
        class_mode=None,batch_size=batch_size, shuffle=False)

    n_validation_files = len(os.listdir(os.path.join(validation_directory, 'validation')))
    bottleneck_features_validation = model.predict_generator(
        generator, n_validation_files/batch_size, verbose=1)
    np.save('../models/bottleneck_features_validation_{}_{}.npy'.format(ts, name), bottleneck_features_validation)

save_bottlebeck_features(size, gen_no_augmentation,
                         train_directory,
                         validation_directory, train_mapping, validation_mapping, name, ts)

print('Done')

print('Training top model...')
def make_top_model(shape):

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy']) # binary crossentropy loss!
    return(model)

def train_top_model(size, training_dir, validation_dir, train_mapping, validation_mapping, name, ts):

    # Load training data: bottleneck features from VGGnet
    train_data = np.load('../models/bottleneck_features_train_{}_{}.npy'.format(ts, name))
    validation_data = np.load('../models/bottleneck_features_validation_{}_{}.npy'.format(ts,name))

    # Mind the ordering!
    train_labels = np.array([train_mapping['train/'+d] for d in os.listdir(os.path.join(training_dir, 'train'))])
    validation_labels = np.array([validation_mapping['validation/'+d] for d in os.listdir(os.path.join(validation_dir, 'validation'))])

    # Define top model
    model = make_top_model(train_data.shape[1:])

    # Use early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1)]
    model.fit(train_data, train_labels,
              epochs=100,
              batch_size=32,
              validation_data=(validation_data, validation_labels), verbose=1,callbacks=callbacks)

    model.save('../models/top_model_{}_{}.h5'.format(ts, name))
    return(train_data.shape)

train_shape = train_top_model(size, train_directory, validation_directory, train_mapping, validation_mapping, name, ts)
print('Done')

print('Reconstruct VGG...')
# See the hero in this thread
# htaps://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

def reconstruct_VGG(top_model_path, size, train_data_shape):
    # build the VGG16 network
    model = k.applications.VGG16(weights='imagenet', include_top=False, input_shape=(size,size,3))
    print('VGG loaded.')

    top_model = make_top_model(shape=model.output_shape[1:])
    top_model.load_weights('../models/top_model_{}_{}.h5'.format(ts, name))

    full_model = Model(inputs=model.input, outputs=top_model(model.output))

    # Set the first 15 layers to non-trainable
    for layer in full_model.layers[:15]:
        layer.trainable=False

    # Compile the model with a SGD/momentum optimizer
    # and a very slow learning rate
    full_model.compile(loss='binary_crossentropy',
                      optimizer=k.optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

    return(full_model)

vgg = reconstruct_VGG('../models/top_model_{}_{}.h5'.format(ts, name),
    size, train_shape)

# Finetune the model
callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
    ModelCheckpoint('../models/VGG_{}_{}.h5'.format(ts, name),
     monitor='val_loss', save_best_only=True, verbose=1)]

vgg.fit_generator(generator=training_generator, steps_per_epoch=n_train_files/batch_size,
                 epochs=epochs, callbacks=callbacks, validation_data=(validation_generator),
                 validation_steps=n_validation_files/batch_size)

print('Done.')
