# coding: utf-8
# Planet_flow with VGGnet according to the keras tutorial
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# -------imports---------- #
######  EARLY STOPPING FOR THE TOP MODEL!!
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

# ---- helper functions ---- #

def save_bottlebeck_features(size, datagen,
    training_dir,
    validation_dir,
    train_mapping,
    validation_mapping, name, ts):

    batch_size=32
    # build the VGG16 network
    model = k.applications.VGG16(include_top=False, weights='imagenet')

    # Training data bottleneck features
    generator = datagen.flow_from_directory(training_dir,
        target_size=(size,size),
        class_mode=None,shuffle=False, batch_size=batch_size)

    n_train_files = len(os.listdir(os.path.join(training_dir, 'train')))
    bottleneck_features_train = model.predict_generator(
        generator, n_train_files/batch_size, verbose=1)
    np.save('../models/bottleneck_features_train_{}_{}.npy'.format(ts, name), bottleneck_features_train)

    # Validation data bottleneck features
    generator = datagen.flow_from_directory(validation_dir,
        target_size=(size,size),
        class_mode=None,batch_size=batch_size, shuffle=False)

    n_validation_files = len(os.listdir(os.path.join(validation_dir, 'validation')))
    bottleneck_features_validation = model.predict_generator(
        generator, n_validation_files/batch_size, verbose=1)
    np.save('../models/bottleneck_features_validation_{}_{}.npy'.format(ts, name), bottleneck_features_validation)

def make_top_model(shape):

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) # binary crossentropy loss!
    return(model)

def train_top_model(size, training_dir, validation_dir, train_mapping, validation_mapping, name, ts, batch_size):
    # Load training data: bottleneck features from VGGnet
    train_data = np.load('../models/bottleneck_features_train_{}_{}.npy'.format(ts, name))
    validation_data = np.load('../models/bottleneck_features_validation_{}_{}.npy'.format(ts,name))

    # Mind the ordering!
    train_labels = np.array([train_mapping['train/'+d] for d in os.listdir(os.path.join(training_dir, 'train'))])
    validation_labels = np.array([validation_mapping['validation/'+d] for d in os.listdir(os.path.join(validation_dir, 'validation'))])

    # Define top model
    model = make_top_model(train_data.shape[1:])

    # Use early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1)]
    model.fit(train_data, train_labels, epochs=100,batch_size=batch_size,validation_data=(validation_data, validation_labels), verbose=1,callbacks=callbacks)

    model.save('../models/top_model_{}_{}.h5'.format(ts, name))
    return(train_data.shape)

def reconstruct_VGG(top_model_path, size, train_data_shape, ts, name):
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
    optimizer = k.optimizers.SGD(lr=1e-4, momentum=0.9)
    full_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    return(full_model, optimizer)

# --- main function ---- #
def save_planet(logger, name, epochs, size, batch_size,
    treshold, iterations, TTA):

    start_time = time.time()

    temp_training_dir, temp_validation_dir = du.make_temp_dirs(logger.ts, name)
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

    # Training data: with TTA - no labels
    # This one has to be calles AFTER the validation data has been moved back!
    # Otherwise it doesn't know the correct amount of images.

    print('Saving bottleneck features...')
    save_bottlebeck_features(size, gen_no_augmentation,
                             train_directory,
                             validation_directory, train_mapping, validation_mapping, name, logger.ts)
    print('Done')

    print('Training top model...')
    train_shape = train_top_model(size, train_directory, validation_directory,
     train_mapping, validation_mapping,  name, logger.ts, batch_size)
    print('Done')

    print('Finetuning VGG...')
    model, optimizer = reconstruct_VGG('../models/top_model_{}_{}.h5'.format(logger.ts, name),
        size, train_shape, logger.ts, name)

    # Finetune the model
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
        ModelCheckpoint('../models/VGG_{}_{}.h5'.format(logger.ts, name),
         monitor='val_loss', save_best_only=True, verbose=1)]

    model.fit_generator(generator=training_generator, steps_per_epoch=n_train_files/batch_size,
                     epochs=epochs, callbacks=callbacks, validation_data=(validation_generator),
                     validation_steps=n_validation_files/batch_size)
    print('Done.')

    # Load best model
    model.load_weights('../models/VGG_{}_{}.h5'.format(logger.ts, name))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # --------move back validation data--------- #
    du.empty_validation_folder(temp_training_dir, temp_validation_dir)

    print('Remapping labels...')
    train_files = [f.split('.')[0] for f in os.listdir(temp_training_dir)]
    train_labels = [df_train.iloc[np.where(df_train.image_name.values == train_file)].tags.values[0] for train_file in train_files]
    y_train = fu.binarize(train_labels, label_map)
    print('Done.')

    n_train_files = len(os.listdir(temp_training_dir))
    # -------Search for best thresholds-------- #
    # Predict full training data. With TTA to make predictions stable!
    print('TTA ({} loops)...'.format(TTA))
    predictions_train = []
    for i in range(TTA):
        training_generator_TTA = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
            class_mode=None, batch_size=batch_size, shuffle=False)
        predictions_train.append(model.predict_generator(generator=training_generator_TTA, steps=n_train_files/batch_size, verbose=1))
    print('Done.')

    # Take the average predicted probabilities
    p_train = np.mean(predictions_train, axis=0)
    print(p_train.shape)

    print('Finding thresholds...')
    from sklearn.metrics import fbeta_score
    best_bac, score = fo.optimize_BAC(y_train, p_train, num_tries=iterations)
    score = str(np.round(score,3))
    score_nothresh = fbeta_score(y_train, (p_train > 0.2).astype(int), beta=2, average='samples')

    print('Score on training data without optimization: {}'.format(score_nothresh))
    print('Score on training data with thresholds bac: {}'.format(score))

    # -------Store-------- #
    if float(score) > treshold:
        print('Test set TTA ({} loops)...'.format(TTA))
        predictions_test = []
        for i in range(TTA):
            # Test data: with TTA - no labels
            test_generator = gen_augmentation.flow_from_directory(test_directory, target_size=(size, size),
                 class_mode=None, batch_size=batch_size, shuffle=False)
            predictions_test.append(model.predict_generator(generator=test_generator, steps=n_test_files/batch_size, verbose=1))
        print('Done')

        p_test = np.mean(predictions_test, axis=0)
        p_test_binary = (p_test > best_bac).astype(int)

        # Convert binary predictions to label strings - define a mapping to the file names
        preds = [' '.join(np.array(labels)[pred_row == 1]) for pred_row in p_test_binary]
        test_mapping = dict(zip([f.split('/')[1].split('.')[0] for f in test_generator.filenames], preds))

        # Map the predictions to filenames in df_test
        predictions_df = pd.DataFrame({'image_name': df_test.image_name, 'tags': df_test.image_name})
        predictions_df['tags'] = predictions_df['tags'].map(test_mapping)

        # Save predictions without consensus predictions
        predictions_df.to_csv('../logs/predictions/VGG_{}_{}_{}.csv'.format(logger.ts, name, score), index=False)

    else:
        logger.log_event('Low score - not storing anything.')

    # Remove temp folders
    du.remove_temp_dirs(logger.ts, name)

    elapsed_time = time.time() - start_time
    print('Elapsed time: {} minutes'.format(np.round(elapsed_time/60, 2)))
    print('Done.')

def main():
    parser = argparse.ArgumentParser(description='Neural network to gain money')
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('epochs', type=int, help="function to execute")
    parser.add_argument('size', type=int, choices=(48,64,96,128,256), help='image size used for training')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='determines batch size')
    parser.add_argument('-t','--treshold', type=float, default=0.85, help='cutoff score for storing models')
    parser.add_argument('-it', '--iterations', type=int, default=100, help='n of iterations for optimizing F score ')
    parser.add_argument('-tta', '--TTA', type=int, default=10, help='number of TTA loops')
    args = parser.parse_args()
    logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

    save_planet(logger,**vars(args))

if __name__ == "__main__":
    sys.exit(main())
