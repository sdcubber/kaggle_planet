# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np 
from spectral import get_rgb, ndvi
import h5py
import time
import cv2
from skimage import io
from tqdm import tqdm


def process_images(data_dir, data_labels, size, extension, process=None):

    data = []

    print('Processing data...')

    for f, tags in tqdm(data_labels.values):
        fpath = os.path.join(data_dir,'{}.{}'.format(f, extension))
        
        img = io.imread(fpath)
        if process == 'NDWI':
            img_layer = get_rgb(img, [3, 2, 1]) # NIR-R-G
            img = (img_layer[:, :, 2] - img_layer[:, :, 0]) / (img_layer[:, :, 2] + img_layer[:, :, 0])
        if process == 'NDVI':
            img = ndvi(img, 2, 3)
        data.append(cv2.resize(img, (size, size)))
        
    
    print('Done')
    
    return data


def pickle_image_data(data, save_dir, tag, size):
    with h5py.File(os.path.join(save_dir, '{}_{}x{}.h5'.format(tag, size, size)), 'w') as hf:
        hf.create_dataset('imgs', data=data)

if __name__ == '__main__':
	
	sys.exit()
