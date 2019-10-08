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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ts_models import lstm

class Metrics(Callback):
	
	def on_train_begin(self, logs={}):
	 self.val_f1s = []
	 self.val_recalls = []
	 self.val_precisions = []
	 self.val_conf_matrix = []
	 
	def on_epoch_end(self, epoch, logs={}):
	 
	 val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round().argmax(axis=1)
	 val_targ = self.validation_data[1].argmax(axis=1)

	 _val_f1 = f1_score(val_targ, val_predict, labels=[0,1], pos_label=1, average='binary')
	 _val_recall = recall_score(val_targ, val_predict, labels=[0,1], pos_label=1, average='binary')
	 _val_precision = precision_score(val_targ, val_predict, labels=[0,1], pos_label=1, average='binary')
	 _val_conf_matrix = confusion_matrix(val_targ, val_predict)

	 self.val_f1s.append(_val_f1)
	 self.val_recalls.append(_val_recall)
	 self.val_precisions.append(_val_precision)
	 
	 print(_val_conf_matrix)
	 
	 return
 
def parse_args():
	parser = argparse.ArgumentParser(description='STEP 03/04 - LSTM training approach using several time-series')
	parser.add_argument("-i", "--series-dir", help='<Required> Input directory that contains the VRT images.', \
		required=True)
	parser.add_argument("-s", "--seed", help='Seed that will be used to split the time-series in train, ' + \
		'validation, test groups. [DEFAULT=2]', type=int, default=2)
	
	parser.add_argument("-n", "--only-evaluate", help='Execute only the evaluation, using the test group.' + \
		' [DEFAULT=False]', action='store_true')
	parser.add_argument("-v", "--validation-split", help='Percentage size of the validation group.' + \
		' [DEFAULT=0.15]', type=float, default=0.15)
	parser.add_argument("-t", "--test-split", help='Percentage size of the test group.' + \
		' [DEFAULT=0.15]', type=float, default=0.15)

	parser.add_argument("-e", "--epochs", help='Number of epochs of the training process. [DEFAULT=100]', \
		type=int, default=100)
	parser.add_argument("-b", "--batch-size", help='Batch size of training process. ', type=int, default=16)
	parser.add_argument("-l", "--learning-rate", help='Learning rate of training process. [DEFAULT=0.00005]', \
		type=float, default=0.00005)

	parser.add_argument("-o", "--output-dir", help='<Required> The output directory that will have the' + \
		' trained model and the tensorboard logs', required=True)
	return parser.parse_args()

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	set_random_seed(seed) 

def evaluate(model_filepath, ts_input, ts_label, identifier):
	e_model = load_model(model_filepath)
	metrics = e_model.evaluate(ts_input, ts_label)
	print("############### " + identifier + " ###############")
	for i in range(0, len(metrics)):
		print("  {}: {}".format(
			e_model.metrics_names[i],
			metrics[i]
		))

def exec(series_dir, output_dir, epochs = 100, batch_size = 15, learning_rate = 0.00005, \
	validation_split = 0.15, test_split = 0.15, only_evaluate = False, seed = 2):

	ts_utils.mkdirp(output_dir)
	
	ts_input, ts_label, ts_info = ts_utils.load_ts_data(series_dir)

	if (test_split > 0 and test_split <= 1):
		ts_input_train, ts_input_test, ts_label_train, ts_label_test = train_test_split(ts_input, ts_label, 
			train_size=(1-test_split), shuffle=True, random_state=seed)
	else:
		ts_input_train = ts_input
		ts_label_train = ts_label

	acc_model_filepath = os.path.join(output_dir, 'acc_model.H5')
	loss_model_filepath = os.path.join(output_dir, 'loss_model.H5')
	last_model_filepath = os.path.join(output_dir, 'last_model.H5')
	
	if not only_evaluate:
		set_seed(seed)

		model = lstm.new_instance(ts_info['input_shape'][1:], learning_rate);
		#print(model.summary())
		
		log_dir = os.path.join(output_dir, 'log')

		callbacks = [
			ModelCheckpoint(last_model_filepath, save_best_only=False),
			ModelCheckpoint(acc_model_filepath, monitor='val_acc', save_best_only=True),
			ModelCheckpoint(loss_model_filepath, monitor='val_loss', save_best_only=True),
			TensorBoard(log_dir=log_dir, write_grads=True, write_images=False),
			Metrics()
		]
		
		model.fit(ts_input_train, ts_label_train, epochs=epochs, batch_size=batch_size, 
			shuffle=True, validation_split=validation_split, callbacks=callbacks
		)

	if (test_split > 0 and test_split <= 1):
		evaluate(last_model_filepath, ts_input_test, ts_label_test, 'LAST MODEL (test)')
		evaluate(acc_model_filepath, ts_input_test, ts_label_test, 'BEST ACC.MODEL (test)')
		evaluate(loss_model_filepath, ts_input_test, ts_label_test, 'BEST LOSS.MODEL (test)')

if __name__ == "__main__":
	args = parse_args()

	series_dir = args.series_dir
	seed = args.seed
	epochs = args.epochs
	batch_size = args.batch_size
	only_evaluate = args.only_evaluate
	learning_rate = args.learning_rate
	validation_split = args.validation_split
	test_split = args.test_split
	output_dir = args.output_dir

	exec(series_dir, output_dir, epochs, batch_size, learning_rate, validation_split, test_split, only_evaluate, seed)
