import os
import time
import numpy as np
import pandas as pd
from keras.models import Sequential

def load_data(filename, normalise_window):
	# filename must be the name of a csv
	df = pd.read_csv(filename, parse_dates=[0], usecols=[0,4])
	result = df.as_matrix()

	if normalise_window:
		result = normalise_windows(result)

	result = np.array(result)

	row = round(0.9 * result.shape[0])
	train = result[:int(row), :]
	np.random.shuffle(train)
	x_train = train[:, :-1]
	y_train = train[:, -1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

	return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
	normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
	normalised_data.append(normalised_window)
	return normalised_datadef normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
	normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
	normalised_data.append(normalised_window)
	return normalised_data

def build_model():
	dropout = 0.5
	conv_size = 9
	conv_stride = 1
	ksize = 2
	pool_stride = 2
	filter_num = 128
	padding = "same"
	model = Sequential()

	# parameters obtained from stock_model.py in Convolutional Neural Stock Market Technical Analyser
	model.add(Conv1D(filter_num, ksize, conv_stride, padding=padding, activation="relu"))
	model.add(Dropout(0.5))
	
	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print("> Compilation Time : ", time.time() - start)
	return model
