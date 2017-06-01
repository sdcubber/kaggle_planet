#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:01:57 2017
Helper functions
@author: stijndc
"""
#------imports----------#

import os
import time
import h5py
import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score

import keras as k
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
#--------------#



def class_weights(y_train, mu=0.50):
    """Compute smooth class weights for imbalanced data
    Based on https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras/16467
    mu is a(nother) tunable parameter.
    Inputs
    ------
    y_train: training data label matrix
    Returns
    ------
    weights_dict: {label_indicator: weight}
    """
    label_dict = {i:np.sum(y_train[:,i]) for i in range(y_train.shape[1])}

    total = np.sum(list(label_dict.values()))
    keys = label_dict.keys()
    weights_dict = dict()

    for key in keys:
        score = math.log(mu*total/float(label_dict[key]))
        weights_dict[key] = score if score > 1.0 else 1.0

    return weights_dict

def load_data(size, extra):
	df_train = pd.read_csv("../data/interim/labels.csv")
	df_test = pd.read_csv("../data/interim/test.csv")
	
	# list of possible labels
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = sorted(list(set(flatten([l.split(' ') for l in df_train['tags'].values]))))

	# Map labels
	label_map = {l: i for i, l in enumerate(labels)}
	inv_label_map = {i: l for l, i in label_map.items()}

	# Read in data from hdf5 dump
	with h5py.File('../data/processed/y_train.h5', 'r') as hf:
		y_train = hf['y_train'][:]

	with h5py.File('../data/processed/train_RGB_{}x{}.h5'.format(size,size), 'r') as hf:
		x_train = hf['imgs'][:]
	print(np.shape(x_train))

	with h5py.File('../data/processed/test_RGB_{}x{}.h5'.format(size,size), 'r') as hf:
		print(hf)
		x_test = hf['imgs'][:]
	if extra:
		with h5py.File('../data/processed/train_NDWI_{}x{}.h5'.format(size,size), 'r') as hf:
			x_train_add = hf['imgs'][:].reshape(-1,size,size,1)
			print(x_train.shape, x_train_add.shape)
			x_train = np.concatenate((x_train, x_train_add),axis=3)
		with h5py.File('../data/processed/test_NDWI_{}x{}.h5'.format(size,size), 'r') as hf:
			x_test_add = hf['imgs'][:].reshape(-1,size,size,1)
			x_test = np.concatenate((x_test, x_test_add),axis=3)
	return labels, df_train, df_test, x_train, y_train, x_test

def normalize_data(x_train, x_valid, x_test):
	x_train_mean = np.mean(x_train)
	x_train_std = np.std(x_train)
	
	x_train_norm = (x_train - x_train_mean)/x_train_std
	x_valid_norm = (x_valid - x_train_mean)/x_train_std
	x_test_norm = (x_test - x_train_mean)/x_train_std
	
	return x_train_norm, x_valid_norm, x_test_norm
	
def resample_data(pos, y, x):
    """Resample features and labels so that they contain an equal amount of positive and negatives
    for the label at position pos
    """

    y_pos = y[y[:,pos] == 1] # Select all the rows that contain the rare label
    x_pos = x[y[:,pos] == 1]

    y_neg = y[y[:,pos] == 0]
    x_neg = x[y[:,pos] == 0]

    n_pos = y_pos.shape[0] # number of positive instances
    print('Number of positive instances: {}'.format(n_pos))
    idx_neg =  np.random.choice(range(y_neg.shape[0]), replace=True, size=int(n_pos*1.5))

    y_resampled = np.concatenate((y_pos, y_neg[idx_neg, :]), axis=0)
    x_resampled = np.concatenate((x_pos, x_neg[idx_neg, :]), axis=0)

    return(y_resampled, x_resampled)


def remap_labels(labels, data, original_labels):
    """Remap some output labels to the correct, alpabetical format"""
    return(pd.DataFrame(data=data, columns=labels).loc[:,original_labels].values)


def log_results(predictions, df_train, df_test, name, timestamp, score):
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
	labels = np.array(sorted(labels))
	print(labels)
	preds = [' '.join(labels[pred_row == 1]) for pred_row in predictions]
	results = pd.DataFrame()
	results['image_name'] = df_test.image_name.values
	results['tags'] = preds
	results.to_csv('../logs/predictions/{}_{}_{}.csv'.format(timestamp, name, score))
