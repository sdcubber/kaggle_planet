# coding: utf-8

# -------imports---------- #

import os
import sys
import time
import argparse
import json
import h5py
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
from sklearn.model_selection import train_test_split



# Import modules
import data.data_utils as du
import features.feature_utils as fu
import models.model_utils as mu
import models.models as m
import models.F_optimizers as fo
import plots.plot_utils as pu
import log_utils as lu


def save_planet(logger, name, epochs, size, batch_size, treshold, debug=False):
	
	# -------load data---------- #
	logger.log_event("Loading data...")
	labels, df_train, df_test, x_train, y_train, x_test = fu.load_data(size)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.10)


	# -------normalize ------- #
	logger.log_event("Preprocessing features...")
	x_min = x_train.min(axis=(0,1,2))
	x_max = x_train.max(axis=(0,1,2))
	x_train, x_valid = fu.normalize(x_train, x_min, x_max), fu.normalize(x_valid, x_min, x_max)
	x_test = fu.normalize(x_test, x_min, x_max)
	
	
	# --------load model--------- #
	logger.log_event("Initializing model...")
	if debug:
		architecture = m.SimpleCNN(size, output_size=len(labels))
	else:
		architecture = m.SimpleNet64_2(size, output_size=len(labels))
	model = Model(inputs=architecture.input, outputs=architecture.output)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1),
				ModelCheckpoint(os.path.join('../models','{}_{}.h5'.format(logger.ts, name)),
				 monitor='val_loss', save_best_only=True, verbose=1)]


	# --------training model--------- #
	logger.log_event("Training model...")
	generator = m.DataAugmenter()
	history = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1,
						batch_size=batch_size, class_weight='auto', validation_data=(x_valid, y_valid),
						callbacks=callbacks)
	generator.fit(x_train)


	# --------log time---------#
	runtime = np.round((time.clock() - logger.start)/60, 2)
	logger.log_event("Running time: {} minutes".format(runtime))


	# ---------Load best model-------------#
	logger.log_event("Load model: {}_{}.h5...".format(logger.ts, name))
	model.load_weights('../models/{}_{}.h5'.format(logger.ts, name))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


	# ---------Making predictions-----------#
	logger.log_event("Predicting training and validation data...")
	# Calculate score on the validation set with the training set-thresholds
	x_full, y_full = np.concatenate((x_train, x_valid), axis=0), np.concatenate((y_train, y_valid), axis=0)
	p_full = model.predict(x_full, verbose=1)
	p_valid = model.predict(x_valid, verbose=1)


	# -------selecting threshold-------- #
	logger.log_event('Finding best threshold...')
	best_anokas = fo.find_thresholds(y_full, p_full, y_valid, p_valid)
	from sklearn.metrics import fbeta_score
	
	logger.log_event('Scoring...')
	score =  fbeta_score(y_valid, p_valid > best_anokas, beta=2, average='samples')
	score = str(np.round(score,3))


	# -------Storing models-------- #
	if float(score) > treshold:
		logger.log_event('Predicting test data...')
		preds = model.predict(x_test, verbose=1)
		p_test_binary = (preds > best_anokas).astype(int)

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
	parser = argparse.ArgumentParser(description='Neural network to gain money')
	parser.add_argument('name', type=str, help="name of your model")
	parser.add_argument('epochs', type=int, help="function to execute")
	parser.add_argument('size', type=int, choices=(32,64,96,128), help='image size used for training')
	parser.add_argument('-b','--batch_size', type=int, default=96, help='determines batch size')
	parser.add_argument('-t','--treshold', type=float, default=0.9, help='cutoff score for storing models')
	parser.add_argument('-db','--debug', action="store_true", help='determines batch size')
	
	args = parser.parse_args()
	logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())
	save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
