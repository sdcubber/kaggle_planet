# coding: utf-8
# Planet_flow with the GFM algorithm on top
# Joint method: matrix P  is constructed with one big network


#########
# TODO: #
#########

# - adapt the generator so that it returns the correct labels -> will this require changes to the generator src code?
# Maybe it works by just adapting the dicts. Keys remain the same, values now become a list of lists instead of a list of ints
# provide weights on loss fields as argument

# -------imports---------- #

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

# Import GFM for the GFM algorithm utility functions
import GFM as GFM
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Import extended datagenerator
# Source: https://gist.github.com/jandremarais/6bf673c76203f612f5ab2981430eb2ef
# See also: https://github.com/fchollet/keras/issues/3296
# !!! The order in which the generator yields the files is not the same order as in the folder structure!
# Though it is the same ordering as listed by os.listdirt(directory)
# This has to do with the way the filesystem orders the files
# Solution: make sure to map the predictions correctly

import extended_generator_GFM

def save_planet(logger, name, epochs, size, batch_size, learning_rate,
    treshold, TTA, debug=False):
    start_time = time.time()
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

    # -------Convert training and validation labels into the matrix format for GFM---------- #
    Y_train = GFM.matrix_Y(y_train)
    Y_valid = GFM.matrix_Y(y_valid)

    # Generate 17 output vectors for training and validation data
    outputs_train = []
    outputs_valid = []
    field_size = []
    enc = encoder.fit(np.arange(0,10).reshape(-1, 1)) # fit the encoder on here and not per label!!! to make sure that every possible class is encoded
    for i in range(Y_train.shape[1]):
    # concatenate train and validation data to fit the encoder
        Y_train_i = enc.transform(Y_train[:,i].reshape(-1, 1))
        Y_valid_i = enc.transform(Y_valid[:,i].reshape(-1, 1))
        outputs_train.append(Y_train_i)
        outputs_valid.append(Y_valid_i)
        field_size.append(Y_train_i.shape[1])

    print('Field sizes: {}'.format(field_size))

    # Put the outputs in a list of lists of arrays
    # (maybe making hte output_train and output_valid lists is redundant?)
    output_fields_train = []
    output_fields_validation = []


    for i in range(n_train_files):
        output_fields_train.append([output_field[i,:] for output_field in outputs_train])

    for i in range(n_validation_files):
        output_fields_validation.append([output_field[i,:] for output_field in outputs_valid])

    # Redefine the labels mappings to the elements of Y_train and Y_valid required for the generators
    # Be careful with the ordering!!!
    # The dicts should have filenames as keys and lists of arrays as values
    print('Redefine label dicts to the elements of matrices P...')
    train_files = [f.split('.')[0] for f in os.listdir(temp_training_dir)]
    val_files = [f.split('.')[0] for f in os.listdir(temp_validation_dir)]

    train_mapping = dict(zip(['train/'+t+'.jpg' for t in train_files], output_fields_train))
    validation_mapping = dict(zip(['validation/'+t+'.jpg' for t in val_files], output_fields_validation))
    print('Done.')

    # Generators - with and without data augmentation
    gen_no_augmentation = extended_generator_GFM.ImageDataGenerator(rescale=1./255)
    gen_augmentation = extended_generator_GFM.ImageDataGenerator(rotation_range=0, width_shift_range=0.05, height_shift_range=0.05,
    horizontal_flip=True, vertical_flip=True, rescale=1./255)

    # Training data: with augmentation - with labels
    training_generator = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
     class_mode='GFM', multilabel_classes=train_mapping, n_class=17, batch_size=batch_size, field_sizes=field_size)

    # Validation data: without augmentation - with labels
    validation_generator = gen_no_augmentation.flow_from_directory(validation_directory, target_size=(size,size),
    class_mode='GFM', multilabel_classes=validation_mapping, n_class=17, batch_size=batch_size, shuffle=False, field_sizes=field_size)

    # --------load model--------- #
    print("Initializing model...")
    if debug:
        architecture = m.SimpleCNN_joint_GFM(size, field_size)
    else:
        architecture = m.SimpleNet64_joint_GFM(size, field_size)

    model = Model(inputs=architecture.input, outputs=architecture.output)

    def lr_schedule(epoch):
        """Learning rate scheduler"""
        return learning_rate * (0.1 ** int(epoch / 10))

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    weights = [1]*17 # put equal weights on all loss fields
    model.compile(loss='binary_crossentropy', optimizer=sgd, loss_weights=weights)

    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
    ModelCheckpoint('../models/GFM_{}_{}.h5'.format(logger.ts, name),
     monitor='val_loss', save_best_only=True, verbose=1),
     LearningRateScheduler(lr_schedule)]

    # --------training model--------- #
    history = model.fit_generator(generator=training_generator, steps_per_epoch=n_train_files/batch_size,epochs=epochs, verbose=1,
    callbacks=callbacks, validation_data=(validation_generator), validation_steps=n_validation_files/batch_size)

    # Load best model
    model.load_weights('../models/GFM_{}_{}.h5'.format(logger.ts, name))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Make predictions for training, for validation and for data. With TTA!
    # Call generators inside the loop!

    print('TTA ({} loops)...'.format(TTA))
    predictions_train = []
    predictions_valid = []
    predictions_test = []
    for i in range(TTA):
        training_generator_TTA = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
            class_mode=None, batch_size=batch_size, shuffle=False)
        validation_generator_TTA = gen_augmentation.flow_from_directory(validation_directory, target_size=(size,size),
            class_mode=None, batch_size=batch_size, shuffle=False)
        test_generator_TTA = gen_augmentation.flow_from_directory(test_directory, target_size=(size,size),
            class_mode=None, batch_size=batch_size, shuffle=False)
        predictions_train.append(model.predict_generator(generator=training_generator_TTA, steps=n_train_files/batch_size, verbose=1))
        predictions_valid.append(model.predict_generator(generator=validation_generator_TTA, steps=n_validation_files/batch_size, verbose=1))
        predictions_test.append(model.predict_generator(generator=test_generator_TTA, steps=n_test_files/batch_size, verbose=1))
    print('Done.')

    # Average TTA predictions - bug here
    def average_TTA(predictions, TTA):
        """Average TTA predictions for GFM output
        Input
        ----
        predictions: list of len TTA
        """

        return([np.mean([np.array(predictions[i][j]) for i in range(TTA)], axis=0) for j in range(17)])

    predictions_train_avg = average_TTA(predictions_train,TTA)
    predictions_valid_avg = average_TTA(predictions_valid,TTA)
    predictions_test_avg = average_TTA(predictions_test,TTA)

    # Fill up the predictions so that they have length 17
    predictions_train_filled = []
    predictions_valid_filled = []
    predictions_test_filled = []

    for pred in predictions_train_avg:
        predictions_train_filled.append(GFM.complete_pred(pred[:,1:], 17))
    for pred in predictions_valid_avg:
        predictions_valid_filled.append(GFM.complete_pred(pred[:,1:], 17))
    for pred in predictions_test_avg:
        predictions_test_filled.append(GFM.complete_pred(pred[:,1:], 17))

    W = GFM.matrix_W_F2(beta=2, n_labels=17)
    (optimal_predictions_train, E_F_train) = GFM.GFM(17, n_train_files, predictions_train_filled, W)
    (optimal_predictions_valid, E_F_valid) = GFM.GFM(17, n_validation_files, predictions_valid_filled, W)
    (optimal_predictions_test, E_F_test) = GFM.GFM(17, n_test_files, predictions_test_filled, W)

    # Store the predictions for the matrix P and the F-optimal predictions to analyze them
    # Also store y_train and the labels
    import pickle

    with open('../logs/pickles/{}_{}_P_train'.format(ts, name), 'wb') as fp:
        pickle.dump(predictions_train_filled, fp)

    with open('../logs/pickles/{}_{}_p_train'.format(ts, name), 'wb') as fp:
        pickle.dump(optimal_predictions_train, fp)

    with open('../logs/pickles/{}_{}_y_train'.format(ts, name), 'wb') as fp:
        pickle.dump(y_train, fp)

    with open('../logs/pickles/{}_{}_filenames'.format(ts, name), 'wb') as fp:
        pickle.dump(training_generator_TTA.filenames, fp)

    score_GFM_train = fbeta_score(y_train, optimal_predictions_train, beta=2, average='samples')
    print('F_2 score on the training data with GFM: {}'.format(score_GFM_train))
    score_GFM_valid = fbeta_score(y_valid, optimal_predictions_valid, beta=2, average='samples')
    print('F_2 score on the validation data with GFM: {}'.format(score_GFM_valid))

    # -------Store-------- #

    if float(score_GFM_valid) > treshold:
        # Convert binary predictions to label strings - define a mapping to the file names
        preds = [' '.join(np.array(labels)[pred_row == 1]) for pred_row in optimal_predictions_test]
        test_mapping = dict(zip([f.split('/')[1].split('.')[0] for f in test_generator_TTA.filenames], preds))

        # Map the predictions to filenames in df_test
        predictions_df = pd.DataFrame({'image_name': df_test.image_name, 'tags': df_test.image_name})
        predictions_df['tags'] = predictions_df['tags'].map(test_mapping)

        # Save predictions without consensus predictions
        predictions_df.to_csv('../logs/predictions/GFM_{}_{}_{}.csv'.format(ts, name, score_GFM_valid), index=False)

    else:
        logger.log_event('Low score - not storing anything.')

def main():
    parser = argparse.ArgumentParser(description='Neural network to gain money')
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('epochs', type=int, help="function to execute")
    parser.add_argument('size', type=int, choices=(32,64,96,128,256), help='image size used for training')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='determines batch size')
    parser.add_argument('-l','--learning_rate', type=float, default=1e-2, help='determines learning rate')
    parser.add_argument('-t','--treshold', type=float, default=0.85, help='cutoff score for storing models')
    parser.add_argument('-db','--debug', action="store_true", help='determines batch size')
    parser.add_argument('-tta', '--TTA', type=int, default=10, help='number of TTA loops')
    args = parser.parse_args()
    logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

    save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
