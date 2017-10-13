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
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


def save_planet(logger, name, epochs, size, batch_size, learning_rate,
				treshold, model, debug):

	# -------load data---------- #
	logger.log_event("Loading data...")
	labels, df_train, df_test, x_train_full, y_train_full, x_test = fu.load_data(size, extra=False)
	if debug:
		print('Debug mode')
		x_train_full, x_test = x_train_full[:100,:,:,:], x_test[:100,:,:,:]
		y_train_full = y_train_full[0:100,:]

	x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=0.10)
	# -------normalize ------- #
	logger.log_event("Preprocessing features...")
	x_train_norm, x_valid_norm, x_test_norm = fu.normalize_data(x_train, x_valid, x_test)

	# --------load model--------- #
	logger.log_event("Initializing model...")

	# create the base pre-trained model
	if model == 'ResNet50':
		base_model = ResNet50(weights='imagenet', include_top=False)

	# add two fully-connected layers
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(512, activation='relu')(x)
	x = Dense(512, activation='relu')(x)

	# and logistic layer
	predictions = Dense(17, activation='sigmoid')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional ResNet50 layers
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='adam', loss='binary_crossentropy')
	callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1),
				ModelCheckpoint('../models/{}_{}.h5'.format(logger.ts, name),
				 monitor='val_loss', save_best_only=True, verbose=1)]

	class_weights=None
	history = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1,
					batch_size=batch_size, class_weight=class_weights, validation_data=(x_valid, y_valid),
					callbacks=callbacks)


	# --------log time---------#
	runtime = np.round((time.clock() - logger.start)/60, 2)
	logger.log_event("Running time: {} minutes".format(runtime))

	# ---------Making predictions-----------#
	logger.log_event("Predicting training and validation data...")
	# Calculate score on the validation set with the training set-thresholds
	p_train = model.predict(x_train_norm, verbose=1)
	p_valid = model.predict(x_valid_norm, verbose=1)

	# -------selecting threshold-------- #
	logger.log_event('Finding best threshold...')
	best_anokas = fo.find_thresholds(y_train, p_train, y_valid, p_valid)

	score =  fbeta_score(y_valid, p_valid > best_anokas, beta=2, average='samples')
	score = str(np.round(score,3))
	logger.log_event('Scoring...')

	# -------Storing models-------- #
	if float(score) > treshold:
		logger.log_event('Predicting test data...')
		p_test = model.predict(x_test_norm, verbose=1)

		p_test_binary = (p_test > best_anokas).astype(int)

		logger.log_event('Generating submission file...')
		fu.log_results(p_test_binary, df_train, df_test, name, logger.ts, score)

		logger.log_event('Saving model architecture and trained weights...')
		# save training history and model architecture
		pd.DataFrame(history.history).to_pickle('../models/{}_{}_{}.pkl'.format(logger.ts, name, score))

		with open('../models/{}_{}_{}_architecture.json'.format(logger.ts, name, score), 'w') as json_file:
			json_file.write(model.to_json())
		logger.log_event('Done!')
	else:
		logger.log_event('Low score - not storing anything.')


def main():
	parser = argparse.ArgumentParser(description='Pretrained models: VGGNet, ResNet...')
	parser.add_argument('name', type=str, help='name of the session')
	parser.add_argument('epochs', type=int, help='number of epochs')
	parser.add_argument('size', type=int, choices=(255,256), help='image size used for training')
	parser.add_argument('-m', '--model', type=str, choices=('ResNet50'), default='ResNet50', help='which pretrained model to use')
	parser.add_argument('-b','--batch_size', type=int, default=30, help='determines batch size')
	parser.add_argument('-l','--learning_rate', type=float, default=1e-3, help='determines learning rate')
	parser.add_argument('-t','--treshold', type=float, default=0.9, help='cutoff score for storing models')
	parser.add_argument('-db','--debug', action="store_true", help='debug mode')

	args = parser.parse_args()
	logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

	save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
