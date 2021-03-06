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

def save_planet(logger, name, epochs, size, batch_size, learning_rate,
				treshold, class_weight, debug=False, extra=None, parallel=False, autoencoder=None):

	# -------load data---------- #
	logger.log_event("Loading data...")
	labels, df_train, df_test, x_train_full, y_train_full, x_test = fu.load_data(size, extra)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=0.10)
	y_train_uncoded = y_train
	y_valid_uncoded = y_valid
	if autoencoder != None:
		_, encoder, decoder = load_autoencoder(autoencoder)
		y_train, y_valid = encoder.predict(y_train), encoder.predict(y_valid)

	# -------normalize ------- #
	logger.log_event("Preprocessing features...")
	x_train_norm, x_valid_norm, x_test_norm = fu.normalize_data(x_train, x_valid, x_test)

	# --------load model--------- #
	logger.log_event("Initializing model...")
	if debug:
		architecture = m.SimpleCNN(size, output_size=len(y_train[0]))
	elif extra != None:
		if parallel:
			architecture = m.SimpleNet64_2_plus_par(size, output_size=len(y_train[0]))
		else:
			architecture = m.SimpleNet64_2_plus(size, output_size=len(y_train[0]))
	else:
		architecture = m.SimpleNet64_2(size, output_size=len(y_train[0]))

	model = Model(inputs=architecture.input, outputs=architecture.output)
	optimizer = Adam(lr=learning_rate)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1),
				ModelCheckpoint('../models/{}_{}.h5'.format(logger.ts, name),
				 monitor='val_loss', save_best_only=True, verbose=1)]


	# --------training model--------- #
	logger.log_event("Training model...")
	generator = m.DataAugmenter()
	if class_weight:
		class_weights = fu.class_weights(y_train)
	else:
		class_weights = None
	history = model.fit(x=x_train_norm, y=y_train, epochs=epochs, verbose=1,
						batch_size=batch_size, class_weight=class_weights, validation_data=(x_valid_norm, y_valid),
						callbacks=callbacks)
	generator.fit(x_train_norm)


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
	p_train = model.predict(x_train_norm, verbose=1)
	p_valid = model.predict(x_valid_norm, verbose=1)

	##########DECODE SEQUENCES##############
	if autoencoder != None:
		p_train = decoder.predict(p_train)
		p_valid = decoder.predict(p_valid)

	# -------selecting threshold-------- #
	logger.log_event('Finding best threshold...')
	best_anokas = fo.find_thresholds(y_train_uncoded, p_train, y_valid_uncoded, p_valid)

	score =  fbeta_score(y_valid_uncoded, p_valid > best_anokas, beta=2, average='samples')
	score = str(np.round(score,3))
	logger.log_event('Scoring...')


	# -------Storing models-------- #
	if float(score) > treshold:
		logger.log_event('Predicting test data...')
		p_test = model.predict(x_test_norm, verbose=1)

		if autoencoder != None:
			p_test = decoder.predict(p_test)

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
	parser = argparse.ArgumentParser(description='Neural network to gain money')
	parser.add_argument('name', type=str, help="name of your model")
	parser.add_argument('epochs', type=int, help="function to execute")
	parser.add_argument('size', type=int, choices=(32,64,96,128), help='image size used for training')
	parser.add_argument('-b','--batch_size', type=int, default=96, help='determines batch size')
	parser.add_argument('-l','--learning_rate', type=float, default=1e-3, help='determines learning rate')
	parser.add_argument('-t','--treshold', type=float, default=0.9, help='cutoff score for storing models')
	parser.add_argument('-w','--class_weight', action="store_true", help='Add class weights relevant to their abundance')
	parser.add_argument('-db','--debug', action="store_true", help='determines batch size')
	parser.add_argument('-X', '--extra', type=str, default=None, choices=(None,'RGB','NDVI','NDWI'), help='add infrared channel')
	parser.add_argument('-p','--parallel', action="store_true", help='use parallel convolutions for extra channel')
	parser.add_argument('-E', '--autoencoder', type=str, default=None, help='Select auto-encoder to train network with')
	args = parser.parse_args()
	logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

	save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
