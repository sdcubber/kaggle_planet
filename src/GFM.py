# coding: utf-8

# -------imports---------- #

import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

# Import modules
import features.feature_utils as fu
import models.model_utils as mu
import models.models as m
import models.F_optimizers as fo
import plots.plot_utils as pu
import log_utils as lu
from labels_auto_encoder import load_autoencoder

from sklearn.metrics import fbeta_score
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)


def matrix_Y(y):
    """Convert binary label matrix to a matrix Y that is suitable to estimate P(y,s):
    Each entry of the matrix Y_ij is equal to I(y_ij == 1)*np.sum(yi)"""
    row_sums = np.sum(y, axis=1)
    Y = np.multiply(y, np.broadcast_to(row_sums.reshape(-1, 1), y.shape)).astype(int)
    return(Y)

def complete_pred(pred, n_labels):
    """Fill up a vector with zeros so that it has lenght 17."""
    if pred.shape[1] < n_labels:
        pred = np.concatenate((pred, np.zeros(shape=(pred.shape[0],n_labels-pred.shape[1]))), axis=1)
    return(pred)

def matrix_W_F2(beta, n_labels):
    """W matrix for F2 measure"""
    W = np.ndarray(shape=(n_labels, n_labels))
    for i in np.arange(1, n_labels+1):
        for j in np.arange(1,n_labels+1):
            W[i-1, j-1] = 1/(i*(beta**2) + j)
    return(W)

def GFM(n_labels, features, predictions, W):
    """GFM algorithm. Implementation according to [1], page 3528.

    Inputs
    -------
    n_labels: n_labels
    features: training features (just the shape is required)
    predictions: list of n_labels nparrays that contain the probabilities required to make up the matrix P
    W: matrix W

    Returns
    ------
    optimal_predictions: F-optimal predictions
    E_f: the expectation of the F-score given x
    """

    E_F = []
    optimal_predictions = []

    for instance in range(features.shape[0]):
        # Construct the matrix P

        P = np.ndarray(shape=(n_labels,n_labels))

        for i in range(n_labels):
            P[i,:] = predictions[i][instance,:]

        # Compute matrix delta
        D = np.matmul(P, W)

        E = []
        h = []

        for k in range(n_labels):
            # solve inner optimization
            h_k = np.zeros(n_labels)
            h_k[np.argsort(D[:,k])[::-1][:k+1]] = 1 # Set h_i=1 to k labels with highest delta_ik
            h.append(h_k)

            # store a value of ...
            E.append(np.dot(h_k,D[:,k]))

        # solve outer maximization problem
        h_F = h[np.argmax(E)]
        E_f = E[np.argmax(E)]

        # Return optimal predictor hF, E[F(Y, hF)]
        optimal_predictions.append(h_F)
        E_F.append(E_f)

    return(np.array(optimal_predictions), E_F)

def planet_GFM(logger, name, epochs, size, method, batch_size, threshold, debug):
    # -------load data ---------- #
    labels, df_train, df_test, x_train_full, y_train_full, x_test = fu.load_data(size, extra=False)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=0.10)

    # Normalize
    x_train_norm, x_valid_norm, x_test_norm = fu.normalize_data(x_train, x_valid, x_test)

    # Convert training labels to the correct format
    Y_train = matrix_Y(y_train)
    Y_valid = matrix_Y(y_valid)

    if method == 'individual':
        ### Matrix P is construced with 17 independent multinomial classifiers

        # Make predictions required for matrix P
        predictions_train_filled = []
        predictions_valid_filled = []
        predictions_test_filled = []
        n_labels = 17

        for i in range(n_labels): # Iterate over the columns of the matrix Y

            # Do a one-hot encoding of column i
            # fit the encoder on all the labels to make sure that every possible class is encoded
            enc = encoder.fit(np.concatenate((Y_train[:,i], Y_valid[:,i]), axis=0).reshape(-1,1))
            Y_train_i = enc.transform(Y_train[:,i].reshape(-1, 1))
            Y_valid_i = enc.transform(Y_valid[:,i].reshape(-1, 1))
            output_size = Y_train_i.shape[1]

            # Multinomial classifier
            if debug:
                architecture = m.SimpleCNN(size, output_size=output_size, output='multiclass')
            else:
                architecture = m.SimpleNet64_2(size, output_size=output_size, output='multiclass')

            print(architecture)
            model = Model(inputs=architecture.input, outputs=architecture.output)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            modelpath = '../models/GFM_single_temp_{}{}.h5'.format(name, logger.ts)

            callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, verbose=1)]

            model.fit(x=x_train_norm, y=Y_train_i, epochs=epochs, verbose=1,
                batch_size=50, validation_data=(x_valid_norm, Y_valid_i),callbacks=callbacks)

            # Load best model to make predictions

            model.load_weights(modelpath)

            pred_train = model.predict(x_train_norm)
            pred_train = pred_train[:,1:] # We don't need the probability that y_i is zero!
            pred_train = complete_pred(pred_train, n_labels)
            predictions_train_filled.append(pred_train)

            pred_valid = model.predict(x_valid_norm)
            pred_valid = pred_valid[:,1:] # We don't need the probability that y_i is zero!
            pred_valid = complete_pred(pred_valid, n_labels)
            predictions_valid_filled.append(pred_valid)

            pred_test = model.predict(x_test_norm)
            pred_test = pred_test[:,1:]
            pred_test = complete_pred(pred_test, n_labels)
            predictions_test_filled.append(pred_test)


    elif method == 'joint':
        # Matrix P is constructed with 1 big network

        # Determine the sizes of the 17 multinomial output fields
        n_labels=17
        field_size = []
        for i in (range(n_labels)):
            enc = encoder.fit(np.concatenate((Y_train[:,i], Y_valid[:,i]), axis=0).reshape(-1,1))
            Y_train_i = enc.transform(Y_train[:,i].reshape(-1, 1))
            field_size.append(Y_train_i.shape[1])
            print(field_size)


        if debug:
            architecture = m.SimpleCNN_joint_GFM(size, field_size)
        else:
            architecture = m.SimpleNet64_joint_GFM(size, field_size)

        model = Model(inputs=architecture.input, outputs=architecture.output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())

        # Generate 17 output vectors for training and validation data
        outputs_train = []
        outputs_valid = []

        for i in range(Y_train.shape[1]):
            enc = encoder.fit(np.concatenate((Y_train[:,i], Y_valid[:,i]), axis=0).reshape(-1,1))
            Y_train_i = enc.transform(Y_train[:,i].reshape(-1, 1))
            Y_valid_i = enc.transform(Y_valid[:,i].reshape(-1, 1))
            outputs_train.append(Y_train_i)
            outputs_valid.append(Y_valid_i)

            modelpath = '../models/GFM_joint_temp_{}{}.h5'.format(name, logger.ts)

            callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, verbose=1)]

        model.fit(x_train_norm, outputs_train, epochs=epochs, verbose=1, callbacks=callbacks,
            batch_size=batch_size, validation_data=(x_valid_norm, outputs_valid))

        predictions_valid = model.predict(x_valid_norm, verbose=1)
        predictions_train = model.predict(x_train_norm, verbose=1)
        predictions_test = model.predict(x_test_norm, verbose=1)

    # Fill up the predictions so that they have length 17
        predictions_train_filled = []
        predictions_valid_filled = []
        predictions_test_filled = []

        for pred in predictions_train:
            predictions_train_filled.append(complete_pred(pred, 17))

        for pred in predictions_valid:
            predictions_valid_filled.append(complete_pred(pred, 17))

        for pred in predictions_test:
            predictions_test_filled.append(complete_pred(pred, 17))

    W = matrix_W_F2(beta=2, n_labels=17)
    (optimal_predictions_train, E_F_train) = GFM(17, x_train_norm, predictions_train_filled, W)
    (optimal_predictions_valid, E_F_valid) = GFM(17, x_valid_norm, predictions_valid_filled, W)
    (optimal_predictions_test, E_F_test) = GFM(17, x_test_norm, predictions_test_filled, W)

    score_GFM_train = fbeta_score(y_train, optimal_predictions_train, beta=2, average='samples')
    print('F_2 score on the training data with multinomial GFM: {}'.format(score_GFM_train))
    score_GFM_valid = fbeta_score(y_valid, optimal_predictions_valid, beta=2, average='samples')
    print('F_2 score on the validation data with multinomial GFM: {}'.format(score_GFM_valid))

    score = score_GFM_valid

    # -------Storing models-------- #
    if float(score) > threshold:
        logger.log_event('Generating submission file...')
        fu.log_results(optimal_predictions_test, df_train, df_test, name, logger.ts, score)
    else:
        logger.log_event('Low score - not storing anything.')

def main():
    parser = argparse.ArgumentParser(description='GFM algorithm')
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('epochs', type=int, help="number of epochs")
    parser.add_argument('size', type=int, choices=(32,64,96,128), help='image size used for training')
    parser.add_argument('method', type=str, choices=('individual', 'joint'), help='method to use for estimating the P matrix')
    parser.add_argument('-b','--batch_size', type=int, default=45, help='determines batch size')
    parser.add_argument('-t','--threshold', type=float, default=0.9, help='cutoff score for storing models')
    parser.add_argument('-db','--debug', action="store_true", help='debug mode')
    args = parser.parse_args()
    logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

    planet_GFM(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
