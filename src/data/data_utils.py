# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np 
import h5py
import time
import cv2
from skimage import io
from tqdm import tqdm


def resize_images(data_dir, data_labels, size):

    data = []

    print('Reading in data...')
    ## Training data

    for f, tags in tqdm(data_labels.values):
        fpath = os.path.join(data_dir,'{}.jpg'.format(f))
        try:
            img = io.imread(fpath)
            data.append(cv2.resize(img, (size, size)))
        except:
            print('Could not load file: {}'.format(fpath))
            break
    
    print('Done')
    
    return data

def pickle_image_data(data, save_dir, tag, size):
    with h5py.File(os.path.join(save_dir, '{}_{}x{}.h5'.format(tag, size, size)), 'w') as hf:
        hf.create_dataset('{}{}x{}'.format(tag, size, size), data=data)

if __name__ == '__main__':
	
	sys.exit()
