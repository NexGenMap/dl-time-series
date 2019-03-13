from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

def new_instance(input_shape, learning_rate):
	
	model = Sequential()
	model.add( LSTM(256, input_shape=input_shape) )
	model.add( Dense(2, activation='softmax') )
	
	optimizer = optimizers.Adam(lr=learning_rate)
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	return model