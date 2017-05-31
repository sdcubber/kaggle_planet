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
        if process == 'RGB':
            img = get_rgb(img, [2,1,0]) # RGB from TIF
            img = np.array(img*128+128, dtype=np.uint8)
        elif process == 'NDWI':
            img_layer = get_rgb(img, [3, 2, 1]) # NIR-R-G
            img = (img_layer[:, :, 2] - img_layer[:, :, 0]) / (img_layer[:, :, 2] + img_layer[:, :, 0])
            img = np.array(img*128+128, dtype=np.uint8)
        elif process == 'NDVI':
            max_value = 0.7210752894  # evaluated on whole dataset
            min_value = -0.71597245876 # evaluated on whole dataset
            img = ndvi(img, 2, 3)
            img = np.array((img-min_value)*256/(max_value-min_value), dtype=np.uint8)
        data.append(cv2.resize(img, (size, size)))
        
    
    print('Done')
    
    return data


def pickle_image_data(data, save_dir, tag, size):
    with h5py.File(os.path.join(save_dir, '{}_{}x{}.h5'.format(tag, size, size)), 'w') as hf:
        hf.create_dataset('imgs', data=data)

if __name__ == '__main__':
	
	sys.exit()
