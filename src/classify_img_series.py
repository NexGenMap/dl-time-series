#!/usr/bin/python3

import numpy as np
import os
import ntpath
from osgeo import gdal
import argparse
import ts_utils
from osgeo import ogr
from datetime import datetime

from sklearn.model_selection import train_test_split

import gdal
import osr
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

def is_outbond(offset, win_size, total_size):
	return (offset + win_size) > total_size

def should_pad(offset, win_size, total_size, pad_size):
	return offset - pad_size >= 0 and (offset + win_size + pad_size) <= total_size

def get_block(img_ds, xoff, yoff, chunk_size, pad_size):
	
	img_xsize = img_ds.RasterXSize 
	img_ysize = img_ds.RasterYSize
	
	win_xsize = win_ysize = chunk_size

	y0_reflect_pad = y1_reflect_pad = 0
	x0_reflect_pad = x1_reflect_pad = 0

	if is_outbond(xoff, win_xsize, img_xsize):
		win_xsize = img_xsize - xoff

	if is_outbond(yoff, win_ysize, img_ysize):
		win_ysize = img_ysize - yoff

	if ((xoff + win_xsize + pad_size) <= img_xsize):
		win_xsize = win_xsize + pad_size
	else:
		x0_reflect_pad = 1

	if (xoff - pad_size >= 0):
		xoff = xoff - pad_size
		win_xsize = win_xsize + pad_size
	else:
		x1_reflect_pad = 1

	if ((yoff + win_ysize + pad_size) <= img_ysize):
		win_ysize = win_ysize + pad_size
	else:
		y0_reflect_pad = 1

	if (yoff - pad_size >= 0):
		yoff = yoff - pad_size
		win_ysize = win_ysize + pad_size
	else:
		y1_reflect_pad = 1

	x = win_xsize + x0_reflect_pad + x1_reflect_pad
	y = win_ysize + y0_reflect_pad + y1_reflect_pad

	block = img_ds.ReadAsArray(xoff, yoff, win_xsize, win_ysize)
	block = np.pad(block, [(0,0), (y0_reflect_pad, y1_reflect_pad), (x0_reflect_pad, x1_reflect_pad)], mode='reflect')

	return block

def create_output_file(base_filepath, out_filepath, raster_count = 1, dataType = gdal.GDT_Float32, \
	imageFormat = 'GTiff', formatOptions = ['COMPRESS=LZW', 'TILED=True', 'BIGTIFF=YES']):
    
  driver = gdal.GetDriverByName(imageFormat)
  base_ds = gdal.Open(base_filepath)

  x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
  x_size = base_ds.RasterXSize 
  y_size = base_ds.RasterYSize
  
  out_srs = osr.SpatialReference()
  out_srs.ImportFromWkt(base_ds.GetProjectionRef())

  output_img_ds = driver.Create(out_filepath, x_size, y_size, raster_count, dataType, formatOptions)
  output_img_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
  output_img_ds.SetProjection(out_srs.ExportToWkt())

  return output_img_ds

def exec(input_dir, model_filepath, output_filepath):

	ts_utils.mkdirp( ts_utils.basedir(output_filepath) )

	images = ts_utils.get_filenames(input_dir, \
		filter_prefix='ts_b', filter_suffix='.vrt')

	model = load_model(model_filepath)

	base_img = images[0]
	base_ds = gdal.Open(base_img)
	n_files = len(images)
	img_xsize = base_ds.RasterXSize 
	img_ysize = base_ds.RasterYSize

	block_size = 256
	pad_size = 1

	out_img_ds = create_output_file(base_img, output_filepath)
	out_band = out_img_ds.GetRasterBand(1)

	count = 0
	nblocks = len(range(0, img_xsize, block_size)) * len(range(0, img_ysize, block_size))

	for xoff in range(0, img_xsize, block_size):
		for yoff in range(0, img_ysize, block_size):
			
			print("Processing block " + str(count) + " of " + str(nblocks) + "("+str(count/nblocks)+"%)")
			print(datetime.now())

			input_ts = []
			for file_idx in range(0, n_files):
				img_ds = gdal.Open(images[file_idx])
				block = get_block(img_ds, xoff, yoff, block_size, pad_size)
				input_ts.append(block)
			input_ts = np.stack(input_ts)

			input_ts = input_ts.transpose((3,2,1,0))
			x, y, time, spectral = input_ts.shape

			input_conv_ts = []

			for xAux in range(1, (x - 1)):
				for yAux in range(1, (y - 1)):
					aux = input_ts[xAux-1:xAux+2, yAux-1:yAux+2, :, :]
					input_conv_ts.append( aux )
			
			input_conv_ts = np.stack(input_conv_ts)
			
			size_x = x - 2*pad_size
			size_y = y - 2*pad_size

			predicted = np.argmax(
					model.predict(input_conv_ts, batch_size=4096), axis=1
			).astype(np.uint16)
			
			result = np.transpose(
				np.reshape(predicted, (size_x, size_y) )
			)

			out_band.WriteArray(result, xoff, yoff)
			count = count + 1

if __name__ == "__main__":
	args = parse_args()

	input_dir = args.input_dir
	model = args.model
	output = args.output

	exec(input_dir, model, output)
