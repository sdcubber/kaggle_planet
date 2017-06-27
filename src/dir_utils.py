# Utility functions to be used with flow_from directory

import os
import numpy as np
import shutil


def make_temp_dirs(ts, name):
    """Make temporary dirs to store the training data.
    Allows to run multiple scripts on HPC simultaneously
    Inputs
    -----
    names, ts: name and timestamp for the temp folders
    """
    temp_training_dir = '../data/TEMP_{}_{}/consensus_train/train'.format(ts, name)
    temp_validation_dir = '../data/TEMP_{}_{}/consensus_validation/validation'.format(ts, name)

    try:
        os.makedirs(temp_training_dir)
        os.makedirs(temp_validation_dir)
    except:
        print('Folders already exist.')

    return(temp_training_dir, temp_validation_dir)


def fill_temp_training_folder(temp_training_dir):
    """Fill temporary training folder with images."""

    for f in os.listdir('../data/interim/consensus_train/train'):
        shutil.copy2(src='../data/interim/consensus_train/train/{}'.format(f),
                     dst=os.path.join(temp_training_dir, '{}'.format(f)))

def move_to_validation_folder(temp_training_dir, temp_validation_dir, fraction=0.10):
    """Randomly move 10% of the training images to the validation folder. """
    n_train = len(os.listdir(temp_training_dir))
    validation_images = np.random.choice(os.listdir(temp_training_dir),size=int(n_train*fraction),replace=False)

    for f in os.listdir(temp_training_dir):
        if f in validation_images:
            shutil.move(src=os.path.join(temp_training_dir, '{}'.format(f)),
                        dst=os.path.join(temp_validation_dir, '{}'.format(f)))
        else:
            continue

def empty_validation_folder(temp_training_dir, temp_validation_dir):
    """Move all the images in the validation folder back to the training folder."""
    for f in os.listdir(temp_validation_dir):
        shutil.move(src=os.path.join(temp_validation_dir, '{}'.format(f)),
                    dst=os.path.join(temp_training_dir, '{}'.format(f)))

def remove_temp_dirs(ts, name):
    """Remove temporary training directories."""
    wd_path = os.getcwd()
    os.chdir('../data') # Move into data folder
    shutil.rmtree('TEMP_{}_{}'.format(ts, name)) # Remove temp folder
    os.chdir(wd_path) # Move back to working directory
