import os
import time
import warnings
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
	# filename must be the name of a csv
	df = pd.read_csv(filename, parse_dates=[0], usecols=[4])
	matrix = df[df.columns[0]].values

	sequence_length = seq_len + 1
	result = []
	for index in range(len(matrix) - sequence_length):
		result.append(matrix[index: index + sequence_length])

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
	return normalised_data

def build_model(layers):
	# parameters obtained from stock_model.py in Convolutional Neural Stock Market Technical Analyser
	dropout = 0.5
	conv_size = 9
	conv_stride = 1
	ksize = 2
	pool_stride = 2
	filter_num = 128
	padding = "same"

	model = Sequential()

	model.add(Conv1D(
		input_shape = (layers[1], layers[0]),
		filters=filter_num, 
		kernel_size=ksize, 
		strides=conv_stride, 
		padding=padding, 
		activation=None))
	model.add(Dropout(0.5))

	model.add(LSTM(
		#input_shape = (layers[1], layers[0]), # (50, 1)
		output_dim = layers[1], # 50
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		layers[2], # 100
		return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(
		output_dim = layers[3])) # 1
	model.add(Activation("linear"))
	
	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print("> Compilation Time : ", time.time() - start)
	return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
