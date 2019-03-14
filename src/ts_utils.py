import os
import gdal
import ntpath
import pickle

import numpy as np

def load_ts_data(input_dir):
	input_filepath = new_filepath('ts_input.npz', directory=input_dir)
	label_filtepath = new_filepath('ts_label.npz', directory=input_dir)
	info_filepath = new_filepath('ts_info.mtd', directory=input_dir)

	print("Reading " + input_filepath)
	ts_input = np.load(input_filepath)
	
	print("Reading " + label_filtepath)
	ts_label = np.load(label_filtepath)
	
	print("Reading " + info_filepath)
	ts_info = load_object(info_filepath)

	return ts_input['data'], ts_label['data'], ts_info

def save_ts_data(ts_input, ts_label, output_dir):
	input_filepath = new_filepath('ts_input.npz', directory=output_dir)
	label_filtepath = new_filepath('ts_label.npz', directory=output_dir)
	info_filepath = new_filepath('ts_info.mtd', directory=output_dir)
	
	ts_info = {
		'input_shape': ts_input.shape,
		'label_shape': ts_label.shape,
	}

	print("Saving " + input_filepath)
	np.savez(input_filepath, data=ts_input)
	
	print("Saving " + label_filtepath)
	np.savez(label_filtepath, data=ts_label)
	
	print("Saving " + info_filepath)
	save_object(info_filepath, ts_info)

def load_object(filepath):
	with open(filepath, 'rb') as file:
		return pickle.load(file)

def save_object(filepath, obj):
	with open(filepath, 'wb') as file:
		pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def mkdirp(output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

def get_nbands(image):
	image_ds = gdal.Open(image)
	return image_ds.RasterCount

def basedir(filepath):
	return os.path.dirname(filepath)

def new_filepath(filepath, suffix=None, ext=None, directory='.'):
	
	filename = ntpath.basename(filepath)

	filename_splited = filename.split(os.extsep)
	filename_noext = filename_splited[0]
	
	if (ext is None):
		ext = filename_splited[1]
	
	if (suffix is not None):
		suffix = '_' + suffix
	else:
		suffix = ''

	filename = filename_noext + suffix + '.' + ext

	return os.path.join(directory, filename)

def get_filenames(input_dir, filter_prefix='', filter_suffix=''):
	
	result = []

	for filename in os.listdir(input_dir):
		if filename.startswith(filter_prefix) and \
			 filename.endswith(filter_suffix):
			result.append(os.path.join(input_dir, filename))

	result.sort()

	return result

def all_images_are_consistent(images):

	for image in images:
		image_ds = gdal.Open(image)
		x_start, pixel_width, _, y_start, _, pixel_height = image_ds.GetGeoTransform()
		x_size = image_ds.RasterXSize 
		y_size = image_ds.RasterYSize
		nbands = image_ds.RasterCount