#!/usr/bin/python3

import numpy as np
import os
import ntpath
from osgeo import gdal
import argparse
import ts_utils
from osgeo import ogr

from sklearn.model_selection import train_test_split

import random
import numpy as np
from tensorflow import set_random_seed

from keras.models import load_model
from rios import applier

def parse_args():
	parser = argparse.ArgumentParser(description='04/04 - Classify image series using a trained model.')
	parser.add_argument("-i", "--input-dir", help='<Required> Input directory that contains the VRTs images.', required=True)
	parser.add_argument("-m", "--model", help='<Required> The model filepath that should ' + \
		' be used in the classification approach.', required=True)
	parser.add_argument("-o", "--output", help='<Required> The output filepath. ' + \
		'The file will be generated in ERDAS_IMG format.', required=True)
	return parser.parse_args()

def exec(input_dir, model_filepath, output_filepath):

	ts_utils.mkdirp( ts_utils.basedir(output_filepath) )

	images = ts_utils.get_filenames(input_dir, \
		filter_prefix='ts_b', filter_suffix='.vrt')

	model = load_model(model_filepath)

	infiles = applier.FilenameAssociations()
	outfiles = applier.FilenameAssociations()
	otherargs = applier.OtherInputs()
	
	otherargs.n_images = len(images)

	for i in range(0, otherargs.n_images):
		key = 'img'+str(i)
		setattr(infiles, key, images[i])
	
	outfiles.result = output_filepath
	otherargs.model = model

	def genmap(info, inputs, outputs, otherargs):    
		size_x, size_y = info.getBlockSize()    
		outputs.result = np.empty((1,size_y,size_x),dtype=np.uint16)

		print("Processing status " + str(info.getPercent()) + "%")
	
		input_ts = []
		for i in range(0, otherargs.n_images):
			key = 'img'+str(i)
			input_img = getattr(inputs, key)
			input_img = np.transpose(input_img)
			
			shape = input_img.shape
			input_img = input_img.reshape(shape[0] * shape[1], shape[2])

			input_ts.append(input_img)

		input_ts = np.dstack(input_ts)
		
		predicted = np.argmax(
				otherargs.model.predict(input_ts), axis=1
		).astype(np.uint16)
		
		outputs.result = np.transpose(
			np.reshape(predicted, (size_x, size_y, 1) )
		)
		
	controls = applier.ApplierControls()
	controls.setNumThreads(1)
	controls.setJobManagerType('multiprocessing')
	applier.apply(genmap, infiles, outfiles, otherargs, controls=controls)

if __name__ == "__main__":
	args = parse_args()

	input_dir = args.input_dir
	model = args.model
	output = args.output

	exec(input_dir, model, output)
