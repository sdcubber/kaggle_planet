# coding: utf-8
# Planet.py, but directly reading in raw data in minibatches
# -------imports---------- #

#########
# TODO: #
#########

# 1.
# - copy training/validation folders to a remote location first!!
# - Otherwise you can only run one script at a time on the cluster...
# - no: just do it with training/validation mappings? -> nope, they have to be in a different folder

# 2.
# - how are images read in? 8 bit integers or 64 bit floats? don't care if it doesnt give memory errors on the cluster

# 3.
# - implement GFM (will require a lot of mapping - remapping)
# - Go over GFM algo to verify its correctness

# 4.
# - Put everything on top of VGGnet (according to the keras tutorial -> fix the mapping once?)

# - (proper scaling of inputs? (does this matter?))
# - (try to implement flow_from_h5py... maybe for later. When using flow_from...,
#   One cpu is reserved anyway to do the loading and the preprocessing so the amount of
#   overhead here is maybe not THAT important)

import os
import sys
import time
import shutil

import argparse
import json
import pandas as pd
import numpy as np
from keras.models import Model
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


def save_planet(logger, name, epochs, size, batch_size, learning_rate,
    treshold, iterations, TTA, debug=False):

    ts = logger.ts

    temp_training_dir, temp_validation_dir = du.make_temp_dirs(ts, name)
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

    # Test data: with TTA - no labels
    test_generator = gen_augmentation.flow_from_directory(test_directory, target_size=(size, size),
     class_mode=None, batch_size=batch_size, shuffle=False)

    # --------load model--------- #
    logger.log_event("Initializing model...")
    if debug:
        architecture = m.SimpleCNN(size, output_size=17)
    else:
        architecture = m.SimpleNet64_2(size, output_size=17)

    model = Model(inputs=architecture.input, outputs=architecture.output)
    optimizer = Adam()

    def lr_schedule(epoch):
        """Learning rate scheduler"""
        return learning_rate * (0.1 ** int(epoch / 10))

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
    ModelCheckpoint('../models/{}_{}.h5'.format(logger.ts, name),
     monitor='val_loss', save_best_only=True, verbose=1),
     LearningRateScheduler(lr_schedule)]

    # --------training model--------- #
    # Load previous weights?

    history = model.fit_generator(generator=training_generator, steps_per_epoch=n_train_files/batch_size,epochs=epochs, verbose=1,
    callbacks=callbacks, validation_data=(validation_generator), validation_steps=n_validation_files/batch_size)

    # Load best model
    model.load_weights('../models/{}_{}.h5'.format(logger.ts, name))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # --------move back validation data--------- #
    du.empty_validation_folder(temp_training_dir, temp_validation_dir)

    print('Remapping labels...')
    train_files = [f.split('.')[0] for f in os.listdir(temp_training_dir)]
    train_labels = [df_train.iloc[np.where(df_train.image_name.values == train_file)].tags.values[0] for train_file in train_files]

    y_train = fu.binarize(train_labels, label_map)
    print('Done.')

    n_train_files = len(os.listdir(temp_training_dir    ))

    # Call the training TTA generator here
    training_generator_TTA = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
        class_mode=None, batch_size=batch_size, shuffle=False)

    # -------Search for best thresholds-------- #
    # Predict full training data. With TTA to make predictions stable!
    print('TTA ({} loops)...'.format(TTA))
    predictions_train = []
    for i in range(TTA):
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
        predictions_df.to_csv('../logs/predictions/{}_{}_{}.csv'.format(logger.ts, name, score), index=False)

        # Save training history and model architecture
        pd.DataFrame(history.history).to_pickle('../models/{}_{}_{}.pkl'.format(logger.ts, name, score))

        with open('../models/{}_{}_{}_architecture.json'.format(logger.ts, name, score), 'w') as json_file:
                json_file.write(model.to_json())
                logger.log_event('Done!')
    else:
        logger.log_event('Low score - not storing anything.')


    # Remove temp folders
    du.remove_temp_dirs(ts, name)

def main():
    parser = argparse.ArgumentParser(description='Neural network to gain money')
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('epochs', type=int, help="function to execute")
    parser.add_argument('size', type=int, choices=(32,64,96,128,256), help='image size used for training')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='determines batch size')
    parser.add_argument('-l','--learning_rate', type=float, default=1e-2, help='determines learning rate')
    parser.add_argument('-t','--treshold', type=float, default=0.85, help='cutoff score for storing models')
    parser.add_argument('-it', '--iterations', type=int, default=100, help='n of iterations for optimizing F score ')
    parser.add_argument('-db','--debug', action="store_true", help='determines batch size')
    parser.add_argument('-tta', '--TTA', type=int, default=10, help='number of TTA loops')
    args = parser.parse_args()
    logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

    save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
