# coding: utf-8
# Planet flow with GFM on top of VGGnet...
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#########
# TODO: #
#########

# - return GFM components for validation and test data
# - return training set predictions
# - check for redundancy in data transformations

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

# ---- helper functions ---- #

def save_bottlebeck_features(size, datagen,
    training_dir,
    validation_dir,
    test_dir, name, ts):

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

    # Test data bottleneck features
    generator = datagen.flow_from_directory(test_dir, target_size=(size,size), class_mode=None, batch_size=batch_size, shuffle=False)

    n_test_files = len(os.listdir(os.path.join(test_dir, 'test')))
    bottleneck_features_test = model.predict_generator(
    generator, n_test_files/batch_size, verbose=1)
    np.save('../models/bottleneck_features_test_{}_{}.npy'.format(ts, name), bottleneck_features_test)

def make_top_model(shape, field_size, nodes):
    architecture = m.GFM_top_classifier(shape=shape, field_sizes=field_size, nodes=nodes)
    model=Model(inputs=architecture.input, outputs=architecture.output)
    model.compile(loss='binary_crossentropy', optimizer='Adam')
    return(model)

def reconstruct_VGG(top_model_path, size, top_model_input_shape, field_size, nodes, ts, name, finetune, optimizer):
    # build the VGG16 network
    model = k.applications.VGG19(weights='imagenet', include_top=False, input_shape=(size,size,3))
    print('VGG loaded.')

    top_model = make_top_model(top_model_input_shape, field_size, nodes)
    top_model.load_weights(top_model_path)

    full_model = Model(inputs=model.input, outputs=top_model(model.output))

    # Set the first 15 layers to non-trainable
    if finetune == 1:
        for layer in full_model.layers[:15]:
            layer.trainable=False
    elif finetune == 2:
        for layer in full_model.layers[:11]:
            layer.trainable=False
    elif finetune == 3:
        # train full model
        print('Training full model')

    # Compile the model with a SGD/momentum optimizer
    # and a very slow learning rate
    if optimizer == 'sgd':
        optimizer = k.optimizers.SGD(lr=1e-5, momentum=0.9)
    else:
        optimizer = k.optimizers.Adam(lr=1e-5)

    full_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer)
    print('VGG reconstructed.')
    return(full_model, optimizer)

def save_planet(logger, name, epochs, size, batch_size, treshold, TTA, nodes, finetune,load_weights,optimizer,debug=False):
    ts=logger.ts
    start_time = time.time()

    # --- Preprocessing --- #
    # Make temporary directories, fill with training and validation data
    temp_training_dir, temp_validation_dir = du.make_temp_dirs(logger.ts, name)
    du.fill_temp_training_folder(temp_training_dir)
    du.move_to_validation_folder(temp_training_dir, temp_validation_dir)

    # High level directories required for generators
    train_directory = os.path.split(temp_training_dir)[0]
    validation_directory = os.path.split(temp_validation_dir)[0]
    test_directory = '../data/interim/consensus_test/'

    n_train_files = len(os.listdir(temp_training_dir))
    n_validation_files = len(os.listdir(temp_validation_dir))
    n_test_files = len(os.listdir(os.path.join(test_directory, 'test')))

    # Load required metadata
    labels, df_train, df_test, label_map, train_mapping, validation_mapping, y_train, y_valid = fu.load_metadata(temp_training_dir, temp_validation_dir)

    #Convert training and validation labels into the matrix format for GFM method

    Y_train = GFM.matrix_Y(y_train)
    Y_valid = GFM.matrix_Y(y_valid)

    # Generate 17 output vectors for training and validation data
    outputs_train = []
    outputs_valid = []
    field_size = []
    enc = encoder.fit(np.arange(0,10).reshape(-1, 1)) # fit the encoder on here and not per label!!! to make sure that every possible class is encoded
    for i in range(Y_train.shape[1]):
        Y_train_i = enc.transform(Y_train[:,i].reshape(-1, 1))
        Y_valid_i = enc.transform(Y_valid[:,i].reshape(-1, 1))
        outputs_train.append(Y_train_i)
        outputs_valid.append(Y_valid_i)
        field_size.append(Y_train_i.shape[1])

    print('Field sizes: {}'.format(field_size)) # Same size everywhere because of fitting prior to for loop

    # Put the outputs in a list of lists of arrays
    # (maybe making the output_train and output_valid lists is redundant?)
    output_fields_train = []
    output_fields_validation = []

    for i in range(n_train_files):
        output_fields_train.append([output_field[i,:] for output_field in outputs_train])

    for i in range(n_validation_files):
        output_fields_validation.append([output_field[i,:] for output_field in outputs_valid])

    # Train and validation mappings required for the generators to yield the labels in the GFM format
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
    horizontal_flip=True, vertical_flip=True, rescale=1./255, fill_mode='reflect')

    # Training data: with augmentation - with labels
    training_generator = gen_augmentation.flow_from_directory(train_directory, target_size=(size,size),
     class_mode='GFM', multilabel_classes=train_mapping, n_class=17, batch_size=batch_size, field_sizes=field_size)

    # Validation data: without augmentation - with labels
    validation_generator = gen_no_augmentation.flow_from_directory(validation_directory, target_size=(size,size),
    class_mode='GFM', multilabel_classes=validation_mapping, n_class=17, batch_size=batch_size, shuffle=False, field_sizes=field_size)

    if debug:
        print('Reading previously stored features...')
        if size == 48:
            features_train = np.load('../models/GFM_VGG/debug_features_train_48.npy')
            features_valid = np.load('../models/GFM_VGG/debug_features_valid_48.npy')
            features_test = np.load('../models/GFM_VGG/debug_features_test_48.npy')
        elif size ==128:
            features_train = np.load('../models/GFM_VGG/debug_features_train_128.npy')
            features_valid = np.load('../models/GFM_VGG/debug_features_valid_128.npy')
            features_test = np.load('../models/GFM_VGG/debug_features_test_128.npy')

    else:
        print('Producing bottleneck features...')
        save_bottlebeck_features(size, gen_no_augmentation,
                                 train_directory,
                                 validation_directory, test_directory, name, logger.ts)
        print('Saving bottleneck features...')
        features_train = np.load('../models/bottleneck_features_train_{}_{}.npy'.format(ts, name))
        features_valid = np.load('../models/bottleneck_features_validation_{}_{}.npy'.format(ts, name))
        features_test = np.load('../models/bottleneck_features_test_{}_{}.npy'.format(ts, name))

    # Rescale these features!
    def min_max_scaling(features, min, max):
        return ((features - min)/(max-min))

    train_max = np.max(features_train)
    train_min = np.min(features_train)

    features_train = min_max_scaling(features_train, train_min, train_max)
    features_valid = min_max_scaling(features_valid, train_min, train_max)
    features_test = min_max_scaling(features_test, train_min, train_max)

    # --- Pretrain the top model --- #
    print('Pretraining top model...')
    model = make_top_model(features_train.shape, field_size, nodes)

    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
    ModelCheckpoint('../models/GFM_top_{}_{}.h5'.format(logger.ts, name),
     monitor='val_loss', save_best_only=True, verbose=1)]

    train_labels = [np.array([output_fields_train[j][i] for j in range(len(features_train))]) for i in range(17)]
    valid_labels = [np.array([output_fields_validation[j][i] for j in range(len(features_valid))]) for i in range(17)]

    history = model.fit(features_train, train_labels, callbacks=callbacks,
     validation_data=(features_valid, valid_labels), epochs=epochs, batch_size=batch_size,
     verbose=1)

    # Store best model
    model.load_weights('../models/GFM_top_{}_{}.h5'.format(logger.ts, name))
    model.compile(loss='binary_crossentropy', optimizer='Adam')
    top_model_path = '../models/top_model_{}_{}.h5'.format(ts, name)
    model.save(top_model_path)
    print('Done.')

    # --- Reconstruct VGG model and finetune top conv layer --- #
    print('Finetuning VGG...')
    model, optimizer = reconstruct_VGG('../models/top_model_{}_{}.h5'.format(logger.ts, name),
        size, features_train.shape, field_size, nodes,logger.ts, name, finetune, optimizer)

    if load_weights:
        print('Loading pretrained weights...')
        # load weights from a previous model run
        assert size == 128, "weights only available for size 128"
        model.load_weights('../models/GFM_VGG/VGG_GFM10072017_17:16_GFM_pre_128.h5')

    # Finetune the model
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
        ModelCheckpoint('../models/VGG_GFM{}_{}.h5'.format(logger.ts, name),
         monitor='val_loss', save_best_only=True, verbose=1),
         ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,cooldown=2,verbose=1)]

    model.fit_generator(generator=training_generator, steps_per_epoch=n_train_files/batch_size,
                     epochs=epochs, callbacks=callbacks, validation_data=(validation_generator),
                     validation_steps=n_validation_files/batch_size)
    print('Done.')

    # Load best model
    model.load_weights('../models/VGG_GFM{}_{}.h5'.format(logger.ts, name))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

    # Average TTA predictions
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

    score_GFM_train = fbeta_score(y_train, optimal_predictions_train, beta=2, average='samples')
    print('F_2 score on the training data with GFM: {}'.format(score_GFM_train))
    score_GFM_valid = fbeta_score(y_valid, optimal_predictions_valid, beta=2, average='samples')
    print('F_2 score on the validation data with GFM: {}'.format(score_GFM_valid))

    # -------Store-------- #
    if float(score_GFM_valid) > treshold:

        # Save test predictions for submission
        # Convert binary predictions to label strings - define a mapping to the file names
        preds = [' '.join(np.array(labels)[pred_row == 1]) for pred_row in optimal_predictions_test]
        test_files = [f.split('.')[0] for f in os.listdir('../data/interim/consensus_test/test/')]
        test_mapping = dict(zip([f for f in test_files], preds))

        # Map the predictions to filenames in df_test
        predictions_df = pd.DataFrame({'image_name': df_test.image_name, 'tags': df_test.image_name})
        predictions_df['tags'] = predictions_df['tags'].map(test_mapping)
        predictions_df.to_csv('../logs/predictions/GFM_VGG_{}_{}_{}.csv'.format(ts, name, score_GFM_valid), index=False)

        # Save training set predictions to optimize ensembling algorithms
        preds = [' '.join(np.array(labels)[pred_row == 1]) for pred_row in optimal_predictions_train]
        train_mapping = dict(zip([f for f in df_train.image_name], preds))

        predictions_df = pd.DataFrame({'image_name': df_train.image_name, 'tags': df_train.image_name})
        predictions_df['tags'] = predictions_df['tags'].map(train_mapping)
        predictions_df.to_csv('../logs/training_predictions/GFM_VGG_{}_{}_{}.csv'.format(ts, name, score_GFM_valid), index=False)

        # Save validation set predictions
        preds = [' '.join(np.array(labels)[pred_row == 1]) for pred_row in optimal_predictions_valid]
        valid_mapping = dict(zip([f for f in df_train.image_name], preds))

        predictions_df = pd.DataFrame({'image_name': df_train.image_name, 'tags': df_train.image_name})
        predictions_df['tags'] = predictions_df['tags'].map(valid_mapping)
        predictions_df.to_csv('../logs/training_predictions/GFM_VGG_validation_{}_{}_{}.csv'.format(ts, name, score_GFM_valid), index=False)

    else:
        logger.log_event('Low score - not storing anything.')

    # Remove temp folders
    du.remove_temp_dirs(logger.ts, name)

    elapsed_time = time.time() - start_time
    print('Elapsed time: {} minutes'.format(np.round(elapsed_time/60, 2)))
    print('Done.')

    # Store the predictions for the matrix P and the F-optimal predictions to analyze them
    # Also store y_train and the labels
    import pickle
    # train
    with open('../logs/pickles/{}_{}_P_train'.format(ts, name), 'wb') as fp:
        pickle.dump(predictions_train_filled, fp)
    with open('../logs/pickles/{}_{}_p_train'.format(ts, name), 'wb') as fp:
        pickle.dump(optimal_predictions_train, fp)
    with open('../logs/pickles/{}_{}_y_train'.format(ts, name), 'wb') as fp:
        pickle.dump(y_train, fp)
    with open('../logs/pickles/{}_{}_filenames'.format(ts, name), 'wb') as fp:
        pickle.dump(training_generator_TTA.filenames, fp)

    # test
    with open('../logs/pickles/{}_{}_P_test'.format(ts, name), 'wb') as fp:
        pickle.dump(predictions_test_filled, fp)
    with open('../logs/pickles/{}_{}_p_test'.format(ts, name), 'wb') as fp:
        pickle.dump(optimal_predictions_test, fp)
    with open('../logs/pickles/{}_{}_filenames_test'.format(ts, name), 'wb') as fp:
        pickle.dump(test_generator_TTA.filenames, fp)

    elapsed_time = time.time() - start_time
    print('Elapsed time: {} minutes'.format(np.round(elapsed_time/60, 2)))

def main():
    parser = argparse.ArgumentParser(description='GFM method on top of VGG16')
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('epochs', type=int, help="function to execute")
    parser.add_argument('size', type=int, choices=(48,64,96,128,224,256), help='image size used for training')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='determines batch size')
    parser.add_argument('-t','--treshold', type=float, default=0.85, help='cutoff score for storing models')
    parser.add_argument('-tta', '--TTA', type=int, default=10, help='number of TTA loops')
    parser.add_argument('-nod', '--nodes', type=int, default=128, help='number of nodes in GFM top model')
    parser.add_argument('-ft', '--finetune', type=int, default=1, help='number of VGG blocks to finetune')
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load pretrained weights')
    parser.add_argument('-db', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-opt', '--optimizer', type=str,
     choices=('sgd', 'adam'),default='sgd', help='optimizer to use for finetuning')
    args = parser.parse_args()
    logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

    save_planet(logger,**vars(args))

if __name__ == "__main__":
    sys.exit(main())
