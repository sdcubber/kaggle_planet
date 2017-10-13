import os
import sys
import argparse
import pandas as pd
import data.data_utils as du
import log_utils as lu

def create_data(data_dir, labels, size, tag, process):
	
	if process ==None:   # temporary quick method
		extension = 'jpg'
	else:
		extension = 'tif'
		
	df_labels = pd.read_csv(labels)
	data = du.process_images(data_dir, df_labels, size, extension, process)
	du.pickle_image_data(data, '../data/processed/',tag, size)

def main():
	parser = argparse.ArgumentParser(description='Process datasets')
	parser.add_argument('data_dir', type=str, help="data folder location")
	parser.add_argument('labels', type=str, help="file containing labels")
	parser.add_argument('size', type=int, choices=(32,64,128,256), help="saving tag")
	parser.add_argument('tag', type=str, help="saving tag")
	parser.add_argument('-p', '--process', type=str, default=None, choices=(None,'RGB','NDWI', 'NDVI'), 
						help="process images")
	args = parser.parse_args()
	create_data(**vars(args))




if __name__ == '__main__':
	
	sys.exit(main())
