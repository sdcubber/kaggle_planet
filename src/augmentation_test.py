# Write away augmented files to understand the augmentation

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

size=256
batch_size=32

# -------load metadata---------- #
labels, df_train, df_test, train_mapping, y_train = fu.load_metadata(consensus_data=True)

# ------ call data generators ------#
gen_train = extended_generator.ImageDataGenerator(rotation_range=10,
 												width_shift_range=0.05, height_shift_range=0.05, fill_mode='reflect',
												horizontal_flip=True, vertical_flip=True, rescale=1./255)

train_directory = '../data/interim/consensus_train/'
n_train_files = len(os.listdir('../data/interim/consensus_train/train'))
train_generator = gen_train.flow_from_directory(train_directory, target_size=(size,size),
class_mode='multilabel', multilabel_classes=train_mapping, n_class=17, batch_size=batch_size, save_to_dir='../data/temp', shuffle=False)

batch_x, batch_y = train_generator.next()
print(batch_y)

# Generator for making predictions - no augmentation!
