#!/usr/bin/python3

import numpy as np
from osgeo import gdal
import argparse
import ts_utils
from osgeo import ogr

from keras.utils import to_categorical

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 02/04 - Generate the time-series \
		considering the vector file informed as samples.')
	parser.add_argument("-i", "--input-dir", help='<Required> Input directory that contains the VRTs images.', required=True)
	parser.add_argument("-s", "--samples", help='<Required> Vector file with the point geometries and class labels.', required=True)
	parser.add_argument("-n", "--num-classes", help='<Required> Number of possible class labels.', required=True, type=int)
	parser.add_argument("-c", "--column-label", help='Name of column that contains the class label values. [DEFAULT=class]', default='class')
	parser.add_argument("-o", "--series-dir", help='<Required> The name of output directory', required=True)
	
	return parser.parse_args()

def exec(input_dir, series_dir, samples_file, num_classes, column_label = 'class'):

	ts_utils.mkdirp(series_dir)
	img_files = ts_utils.get_filenames(input_dir, \
		filter_prefix='ts_b', filter_suffix='.vrt')

	ds_images = []
	for img_file in img_files:
		ds_images.append(gdal.Open(img_file))

	gt_image = ds_images[0].GetGeoTransform()

	ds_samples = ogr.Open(samples_file)
	features = ds_samples.GetLayer()
	
	ts_input = []
	ts_label = []
	
	print("Reading data from", img_files)
	for feature in features:
		label = feature.GetField('class')
		
		point = feature.GetGeometryRef()
		coord_x, coord_y = point.GetX(),point.GetY()
		
		xoff = int((coord_x - gt_image[0]) / gt_image[1]) #x pixel
		yoff = int((coord_y - gt_image[3]) / gt_image[5]) #y pixel
		
		feature_input = []
		feature_label = to_categorical(label, num_classes)
		
		for ds_image in ds_images:			
			tseries = ds_image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=1, ysize=1)
			nbands, _, _ = tseries.shape
			
			feature_input.append(tseries.reshape((1, nbands)))

		ts_input.append(np.vstack(feature_input))
		ts_label.append(feature_label)

	ts_input = np.dstack(ts_input).transpose((2,1,0))
	ts_label = np.stack(ts_label)

	print('Time-series input shape:', ts_input.shape)
	print('Time-series label shape:', ts_label.shape)

	ts_utils.save_ts_data(ts_input, ts_label, series_dir)

	return ts_input, ts_label

if __name__ == "__main__":
	args = parse_args()

	input_dir = args.input_dir
	samples = args.samples
	num_classes = args.num_classes
	column_label = args.column_label
	series_dir = args.series_dir

	exec(input_dir, series_dir, samples, num_classes, column_label)
