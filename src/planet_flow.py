# coding: utf-8
# Planet.py, but directly reading in raw data in minibatches
# -------imports---------- #

#########
# TODO: #
#########

# + implement best threshold search from bac
# + check if necessary libraries are on cluster: check (PIL, sk-image, openCV)
# - add sufficient dropout so that early stopping is not needed?
# - make predictions for test set, concat them with consensus predictions -> write some separate code to do the concatenation
# - try to implement flow_from_h5py...

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

# Import extended datagenerator
# Source: https://gist.github.com/jandremarais/6bf673c76203f612f5ab2981430eb2ef
import extended_generator

def save_planet(logger, name, epochs, size, batch_size, learning_rate,
				treshold, debug=False):

	# -------load metadata---------- #
	labels, df_train, df_test, train_mapping, y_train = fu.load_metadata(consensus_data=True)

	# ------ call data generators ------#

	# Generator for training images - with data augmentation
	gen_train = extended_generator.ImageDataGenerator(rotation_range=20, width_shift_range=0.05, height_shift_range=0.05,
												horizontal_flip=True, vertical_flip=True, rescale=1./255)

	train_directory = '../data/interim/consensus_train/'
	n_train_files = len(os.listdir('../data/interim/consensus_train/train'))
	train_generator = gen_train.flow_from_directory(train_directory, target_size=(size,size),
	 class_mode='multilabel', multilabel_classes=train_mapping, n_class=17, batch_size=batch_size)

	# Generator for making predictions - no augmentation!
	gen_pred_train = extended_generator.ImageDataGenerator(rescale=1./255)
	train_generator_predictions = gen_pred_train.flow_from_directory(train_directory, target_size=(size,size), class_mode='multilabel',
	 						multilabel_classes=train_mapping, n_class=17, batch_size=batch_size)

	test_gen = extended_generator.ImageDataGenerator(rescale=1./255)
	test_directory = '../data/interim/consensus_test/'
	n_test_files = len('../data/interim/consensus_test/test')
	test_generator_predictions = test_gen.flow_from_directory(test_directory, target_size=(size, size), class_mode=None, batch_size=batch_size)

	# --------load model--------- #
	logger.log_event("Initializing model...")
	if debug:
		architecture = m.SimpleCNN(size, output_size=17)
	else:
		architecture = m.SimpleNet64_2(size, output_size=len(y_train[0]))

	model = Model(inputs=architecture.input, outputs=architecture.output)
	optimizer = Adam(lr=learning_rate)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1),
				ModelCheckpoint('../models/{}_{}.h5'.format(logger.ts, name),
				 monitor='val_loss', save_best_only=True, verbose=1)]

	# --------training model--------- #
	history = model.fit_generator(generator=train_generator, steps_per_epoch=n_train_files/batch_size,epochs=epochs, verbose=1)

	# ---------Making predictions-----------#
	# Calculate score on the validation set with the training set-thresholds
	p_train = model.predict_generator(generator=train_generator_predictions, steps=n_train_files/batch_size, verbose=1)
	# -------search for best thresholds-------- #

	from sklearn.metrics import fbeta_score
	best_bac, score = fo.optimize_BAC(y_train, p_train, num_tries=20)
	score = str(np.round(score,3))
	score_nothresh = fbeta_score(y_train, (p_train > 0.2).astype(int), beta=2, average='samples')
	print('Score on trainin data without optimization: {}'.format(score_nothresh))
	print('Score on training data: {}'.format(score))

	# -------Store-------- #
	if float(score) > treshold:
		p_test = model.predict_generator(generator=batches_test, steps=n_test_files/batch_size, verbose=1)
		p_test_binary = (p_test > best_bac).astype(int)

		fu.log_results(p_test_binary, df_train, df_test, name, logger.ts, score)

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
	parser.add_argument('size', type=int, choices=(32,64,96,128,256), help='image size used for training')
	parser.add_argument('-b','--batch_size', type=int, default=96, help='determines batch size')
	parser.add_argument('-l','--learning_rate', type=float, default=1e-3, help='determines learning rate')
	parser.add_argument('-t','--treshold', type=float, default=0.9, help='cutoff score for storing models')
	parser.add_argument('-db','--debug', action="store_true", help='determines batch size')
	args = parser.parse_args()
	logger = lu.logger_class(args, time.strftime("%d%m%Y_%H:%M"), time.clock())

	save_planet(logger,**vars(args))


if __name__ == "__main__":
    sys.exit(main())
