#!/usr/bin/python3

# Copyright (C) 2018	Evandro C Taquary
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.	

SEED=0 #an integer acting as a randomization parameter for data shuffling and model initializing.

import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
from tensorflow import set_random_seed                                                                                                                                                                                        
set_random_seed(SEED) 

from osgeo import gdal, ogr
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os.path
import sys
from sklearn.model_selection import train_test_split
from shutil import copyfile
from rios import applier

def print_usage():
    print("Usage: {} <MODE> data_path".format(sys.argv[0]))
    print("Avaiable modes: train, eval, predict")
    print("For train mode check parameters in the code.")
    exit(0)

def get_point_idx(fname_shp, fname_geo):    
    """
    Take the geotransform parameters of fname_geo and perform an affine geotransform 
    of the point's coordinates within fname_shp shapefile. 
    Return a list of pixels' indices of the corresponding matrix.
    TODO: let the user choose whether fname_geo is a GeoTIFF image itself or a GeoTransform object.
    """
    ds_geo = gdal.Open(fname_geo)
    gt = ds_geo.GetGeoTransform()
    ds_shp = ogr.Open(fname_shp)
    layer = ds_shp.GetLayer()
    pts_idxs = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        mx,my = geom.GetX(),geom.GetY()
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel
        pts_idxs.append((px,py))
    return pts_idxs    

def fetch_time_series(fname_raster, fname_shp):
    """
    Take the points of fname_shp and gather the corresponding time series from fname_raster.
    Return a matrix with all timeseries where each row holds the pixels' values through time.
    """
    pts_idxs = get_point_idx(fname_shp, fname_raster)
    ds_band = gdal.Open(fname_raster)
    n_images = ds_band.RasterCount
    n_pts = len(pts_idxs)
    matrix = np.empty(shape=(n_pts,n_images))
    for i,(px,py) in enumerate(pts_idxs):
        tseries = ds_band.ReadAsArray(xoff=px, yoff=py, xsize=1, ysize=1)
        matrix[i,:] = np.reshape(tseries,(1,n_images))
    return matrix

def load_for_keras(*fname_raster, **fname_class):
    """
    Generate the time-cube tensor with all the timeseries within *fname_raster files, corresponding 
    to the points within each class of **fname_class files. Also generate the matrix with corresponding  
    reference labels in One Hot Encoding formart.
    Return the tensors with shapes ready to use with keras.layers.LSTM. Position of each class 
    at the One Hot Encoding vector will coincide to the order in which it appears at **fname_classe.
    TODO: if the user names one class with an integer <= # of classes, use that integer to identify the class
    TODO: return also a mapping of the class id to the class name
    TODO: let user choose whether returned values shall be shuffled or not 
    """
    bands = []
    for band in fname_raster:
        tseries = []
        for fname_shp in fname_class.values():
            current = fetch_time_series(band,fname_shp)
            tseries.append(current)
        bands.append(np.vstack(tseries))
    X = np.dstack(bands)
    labels = []
    total = 0
    for i,ts in enumerate(tseries): #use last retrieved tseries to get # of points of each class (assuming they have same # of points)
        total += len(ts)
        one_hot = [to_categorical(i,len(fname_class))]*len(ts)
        labels.append(one_hot)
    Y = np.vstack(labels)
    return X,Y

def load_db(lst_raster, lst_class, train_size=1.0, shuffle=True, random_state=None, cache=None):    
    """
    Load a time-series dataset and format it for training and/or evaluating. Optionally store it in a cache file aiming 
    performance, persistence and reproducibility. If the value of cache is None, no cache file is created. lst_raster receives a list
    of raster files: each one holds a unique spectral band for all images of the time-series. lst_class is a list of multipoint shapefiles
    in which each of the them represents a diferent LCLU class.
    Return either a tensor containing all time-serieses and its corresponding labels tensor, both partitioned into two subsets. The number 
    of samples in each partition depends on the value of train_size, which is the proportion used to split the dataset. If shuffle
    is set to True, dataset is going to be shuffled before splited. The randomness of shuffle will be determined by random_state.
    """
    if cache is not None:
        try:
            file = np.load(cache)
            X_train,Y_train,X_test,Y_test = file['X_train'],file['Y_train'],file['X_test'],file['Y_test']
            file.close()
            print("Cache file has been read.")
        except FileNotFoundError:
            print("No cache file, creating a new one from provided database...")
            X,Y = load_for_keras(*lst_raster, **lst_class)
            X_train,X_test,Y_train,Y_test = train_test_split(X, Y, train_size=train_size, shuffle=shuffle, random_state=random_state)
            try:
                np.savez(cache,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
            except OSError: #TODO: specify error
                print("Warning: database loaded but no cache file created!")
    else:
        print("No cache file provided, fetching dataset from source.")
        X,Y = load_for_keras(*lst_raster, **lst_class)
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    return (X_train,X_test,Y_train,Y_test)

def train(X_train, Y_train, epochs=200, log_dir=None, lmodel=None, bmodel_acc=None, bmodel_loss=None):
    """
    Perform training of the RNN model where X_train is the input time-serieses tensor, Y_train it's correspondent labels, epochs is the number
    of epochs used in training; lmodel, the path to store file containing the model state after training all epochs; bmodel_acc,
    the path to store the model state after the epoch which yielded the best accuracy; and, bmodel_loss, analogous to the last
    but considering the lowest loss.
    Return the trained model.
    """
    model = Sequential()
    model.add(LSTM(256,input_shape=(X_train.shape[1:])))
    model.add(Dense(2, activation='softmax'))
    adam = optimizers.Adam(lr=0.00002)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    callbacks=[]
    if lmodel is not None:
        callback_last = ModelCheckpoint(lmodel, save_best_only=False)
        callbacks.append(callback_last)
    if bmodel_acc is not None:
        callback_acc = ModelCheckpoint(bmodel_acc, monitor='val_acc', save_best_only=True)
        callbacks.append(callback_acc)
    if bmodel_loss is not None:
        callback_acc = ModelCheckpoint(bmodel_loss, monitor='val_loss', save_best_only=True)
        callbacks.append(callback_acc)
    if log_dir is not None:
        callback_tb = TensorBoard(log_dir=log_dir,write_grads=True, write_images=False)
        callbacks.append(callback_tb)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=16, shuffle=True, validation_split=0.15, callbacks=callbacks)  #TODO: investigate shuffle option; implement ModelCheckpoint callback
    return model

def evaluate(X,Y,model):
    score = model.evaluate(X, Y)
    return score

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print_usage()
mode = sys.argv[1]
data_path = "./data/" if len(sys.argv)==2 else sys.argv[2]
if mode not in ['train','predict','evaluate','eval']:
    print_usage()
if not os.path.isdir(data_path):
    print("\nError: provided directory '{}' doesn't exist.\n".format(data_path))
    print_usage()

###CACHE FILE WHERE TSERIES ARE STORED FOR PERSTENCE###
fname_cache=data_path+"t_series.npz"
###BANDS' FILES###
fname_blue = data_path+"blue.vrt"
fname_green = data_path+"green.vrt"
fname_red = data_path+"red.vrt"
fname_nir = data_path+"nir.vrt"
fnames_rasters = (fname_blue, fname_green, fname_red, fname_nir)
###CLASSES' SHAPE FILES###
fname_pasture = data_path+"points_pasture.shp"
fname_nonpast = data_path+"points_nonpast.shp"
fnames_classes = {'pasture':fname_pasture, 'nonpast':fname_nonpast}
###MODEL FILE###
fname_last = data_path+"model_last.h5"
fname_acc = data_path+"model_acc.h5"
fname_loss = data_path+"model_loss.h5"
###TENSORBOARD LOG DIR###
log_dir = data_path+"log/"
###FILE HOLDING THE ENTIRE SCRIPT USED FOR A TRAINING INSTANCE###
fname_used = data_path+"used_script.py"
###CURRENT SCRIPT'S PATH###
fname_script = os.path.realpath(__file__)
###CLASSIFIED IMAGE'S PATH###
fname_output = data_path+"classified.img"

X_train,X_test,Y_train,Y_test = load_db(fnames_rasters, fnames_classes, train_size=.85, shuffle=True, random_state=SEED, cache=fname_cache)

if mode == 'train':
    model = train(X_train,Y_train, log_dir=log_dir, lmodel=fname_last, bmodel_acc=fname_acc, bmodel_loss=fname_loss)
    print("-"*150)
    print("Test dataset evaluation:\n")
    print("### LAST EPOCH MODEL ###")
    score_test = evaluate(X_test,Y_test,model)        
    print("Test dataset scores -> {}: {} - {}: {}".format(model.metrics_names[1],score_test[1]*100, model.metrics_names[0],score_test[0]))
    try:
        copyfile(fname_script, fname_used)
    except OSError:
        print("Warning: the script used to generate the model couldn't be saved onto datasource.")
elif mode=='predict':
    try:
        model = load_model(fname_acc)
        print("Loaded model from disk.")
    except OSError:
        print("Error: cannot load model's files! Have you ever perform training?")
        exit(0) 
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    infiles.blue = fname_blue
    infiles.green = fname_green
    infiles.red = fname_red
    infiles.nir = fname_nir
    outfiles.map = fname_output
    otherargs.model = model

    def genmap(info, inputs, outputs, otherargs):    
        size_x, size_y = info.getBlockSize()    
        outputs.map = np.empty((1,size_y,size_x),dtype=np.uint16)
        inputs.blue = np.transpose(inputs.blue)
        inputs.blue = inputs.blue.reshape(inputs.blue.shape[0]*inputs.blue.shape[1],inputs.blue.shape[2])
        inputs.green = np.transpose(inputs.green)
        inputs.green = inputs.green.reshape(inputs.green.shape[0]*inputs.green.shape[1],inputs.green.shape[2])
        inputs.red = np.transpose(inputs.red)
        inputs.red = inputs.red.reshape(inputs.red.shape[0]*inputs.red.shape[1],inputs.red.shape[2])
        inputs.nir = np.transpose(inputs.nir)
        inputs.nir = inputs.nir.reshape(inputs.nir.shape[0]*inputs.nir.shape[1],inputs.nir.shape[2])
        tensor = np.dstack([inputs.blue,inputs.green,inputs.red,inputs.nir])
        predict = np.argmax(otherargs.model.predict(tensor),axis=1).astype(np.uint16)
        outputs.map = np.transpose(np.reshape(predict,(size_x,size_y,1)))
        print("Processing status " + str(info.getPercent()) + "%")

    controls = applier.ApplierControls()
    controls.setNumThreads(1) #TODO: currently works only with one thread, fix it
    controls.setJobManagerType('multiprocessing')
    applier.apply(genmap, infiles, outfiles, otherargs, controls=controls)
elif mode=='evaluate' or mode=='eval':   
    try:
        model_acc = load_model(fname_acc)
        print("\n### BEST ACCURACY MODEL ###")
        score_test = evaluate(X_test,Y_test,model_acc)        
        print("Test dataset scores -> {}: {} - {}: {}".format(model_acc.metrics_names[1],score_test[1]*100, model_acc.metrics_names[0],score_test[0]))
        score_train = evaluate(X_train,Y_train,model_acc)
        print("Train dataset scores -> {}: {} - {}: {}\n".format(model_acc.metrics_names[1],score_train[1]*100, model_acc.metrics_names[0],score_train[0]))
    except OSError:
        print("Error: cannot load best accuracy model! Have you ever perform training?")
    try:
        model_loss = load_model(fname_loss)
        print("### BEST LOSS MODEL ###")
        score_test = evaluate(X_test,Y_test,model_loss)        
        print("Test dataset scores -> {}: {} - {}: {}".format(model_loss.metrics_names[1],score_test[1]*100, model_loss.metrics_names[0],score_test[0]))
        score_train = evaluate(X_train,Y_train,model_loss)
        print("Train dataset scores -> {}: {} - {}: {}\n".format(model_loss.metrics_names[1],score_train[1]*100, model_loss.metrics_names[0],score_train[0]))
    except OSError:
        print("Error: cannot load best loss model! Have you ever perform training?")
    try:
        model_last = load_model(fname_last)
        print("### LAST EPOCH MODEL ###")
        score_test = evaluate(X_test,Y_test,model_last)        
        print("Test dataset scores -> {}: {} - {}: {}".format(model_last.metrics_names[1],score_test[1]*100, model_last.metrics_names[0],score_test[0]))
        score_train = evaluate(X_train,Y_train,model_last)
        print("Train dataset scores -> {}: {} - {}: {}\n".format(model_last.metrics_names[1],score_train[1]*100, model_last.metrics_names[0],score_train[0]))
    except OSError:
        print("Error: cannot load last epoch model! Have you ever perform training?")
else:
    print('Invalid mode!')
    print_usage()