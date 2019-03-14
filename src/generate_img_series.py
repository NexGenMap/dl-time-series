#!/usr/bin/python3

import os
import ntpath
import gdal
import argparse
import ts_utils

import subprocess

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 01/04 - Stack all images of input directory, \
		producing one Virtual Dataset-VRT per band in output directory')
	parser.add_argument("-i", "--input-dir", help='<Required> Input image directory.', required=True)
	parser.add_argument("-b", "--bands", nargs='+', type=int, help='The bands that should be considered. [DEFAULT=All]', default=None)
	parser.add_argument("-o", "--output-dir", help='<Required> Output VRTs directory', required=True)
	return parser.parse_args()

def create_vrt_bands(img_path, output_vrt, bands):
	
	image_ds = gdal.Open(img_path, gdal.GA_ReadOnly)

	vrt_bands = []
	if bands is None:
		bands = range(1, (image_ds.RasterCount+1) )

	for band in bands:
		vrt_filepath = ts_utils.new_filepath(img_path, suffix = str(band), ext='vrt', 
			directory=ts_utils.basedir(output_vrt))

		command = ["gdalbuildvrt"]
		command += ["-b", str(band)]
		command += [vrt_filepath]
		command += [img_path]
		
		subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		vrt_bands += [vrt_filepath]

	return vrt_bands

def create_separate_bands(images, output_vrt, bands):
	separate_bands = []

	for img_path in images:
		vrt_bands = create_vrt_bands(img_path, output_vrt, bands)
		separate_bands += vrt_bands

	return separate_bands

def create_vrt_output(input_imgs, output_vrt, bands = None):
	separate_bands = create_separate_bands(input_imgs, output_vrt, bands)

	command = ["gdalbuildvrt"]
	command += ["-separate"]

	command += [output_vrt]
	command += separate_bands

	print('Creating vrt file ' + output_vrt)

	subprocess.call(command, stdout=subprocess.PIPE)

def exec(input_dir, output_dir, bands = None):
	ts_utils.mkdirp(output_dir)
	images = ts_utils.get_filenames(input_dir, filter_suffix='.tif')

	if bands is None:
		nbands = ts_utils.get_nbands(images[0])
		bands = bands = range(1, (nbands+1) )
	
	for band in bands:
		output_vrt = os.path.join(output_dir, 'ts_b' + str(band) + '.vrt')
		create_vrt_output(images, output_vrt, [band])

if __name__ == "__main__":
	args = parse_args()

	input_dir = args.input_dir
	bands = args.bands
	output_dir = args.output_dir

	exec(input_dir, output_dir, bands)