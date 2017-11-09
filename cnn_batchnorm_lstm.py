import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def build_model(layers, cnn_layers):
	# parameters obtained from stock_model.py in Convolutional Neural Stock Market Technical Analyser
	dropout = 0.5
	conv_stride = 1
	ksize = 5
	pool_stride = 2
	filter_num = 128
	padding = "same"
	lstm_units = 64

	model = Sequential()

	model.add(Conv1D(
		input_shape = (layers[1], layers[0]), # (50, 1)
		filters=filter_num, 
		kernel_size=ksize, 
		strides=conv_stride, 
		padding=padding, 
		activation=None))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))
	
	model.add(MaxPooling1D(
		pool_size=2))

	for x in range(1, cnn_layers):
		model.add(Conv1D(
			filters=filter_num*2*x, 
			kernel_size=ksize, 
			strides=conv_stride, 
			padding=padding, 
			activation=None))
		BatchNormalization(axis=-1)
		model.add(Activation('relu'))
		
		model.add(MaxPooling1D(
			pool_size=2))

	model.add(LSTM(
		#output_dim = layers[1], # 50
		lstm_units,
		return_sequences=True))
	model.add(Dropout(dropout/2))

	model.add(LSTM(
		#layers[1]*2, # 100
		lstm_units,
		return_sequences=False))
	model.add(Dropout(dropout))

	model.add(Dense(
		output_dim = 1)) # Linear output
	model.add(Activation("linear"))
	
	start = time.time()
	model.compile(loss="mse", optimizer="adadelta")
	print("> Compilation Time : ", time.time() - start)
	return model
