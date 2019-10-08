from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Conv3D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Cropping3D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import Input

from keras import optimizers

def new_instance(input_shape, learning_rate):
	
	x, y, time, spectral = input_shape

	inputLayer = Input(shape=input_shape)

	cnn_model = Conv3D(256, kernel_size=(3, 3, 5), padding='same')(inputLayer)
	cnn_model = Conv3D(256, kernel_size=(3, 3, 1), padding='valid')(cnn_model)
	cnn_model = BatchNormalization()(cnn_model)
	cnn_model = Activation('relu')(cnn_model)
	cnn_model = Flatten()(cnn_model)

	lstm_model = Cropping3D(cropping=(1, 1, 0))(inputLayer)
	lstm_model = Reshape(target_shape=(time, spectral))(lstm_model)
	lstm_model = BatchNormalization()(lstm_model)
	lstm_model = Bidirectional(CuDNNLSTM(256, return_sequences=True))(lstm_model)
	lstm_model = Flatten()(lstm_model)

	conc_model = concatenate([lstm_model, cnn_model])
	conc_model = Dense(256, activation='relu')(conc_model)
	conc_model = Dropout(0.3)(conc_model)
	conc_model = Dense(64, activation='relu')(conc_model)
	conc_model = Dense(2, activation='sigmoid')(conc_model)
	
	conc_model = Model(inputLayer, conc_model)
	optimizer = optimizers.Nadam(lr=learning_rate)
	conc_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	return conc_model