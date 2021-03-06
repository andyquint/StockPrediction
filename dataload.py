import numpy as np
import pandas as pd
from numpy import newaxis
import math
from scipy.signal import savgol_filter

def load_data(filename, seq_len, normalise_window=True, smoothing=False, smoothing_window_length=5, smoothing_polyorder=3):
    # filename must be the name of a csv
    df = pd.read_csv(filename, parse_dates=[0], usecols=[4])
    matrix = df[df.columns[0]].values

    if (smoothing):
        matrix = savgol_filter(matrix, window_length=smoothing_window_length, polyorder=smoothing_polyorder)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(matrix) - sequence_length):
        result.append(matrix[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.seed(100)
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def load_sin_data(seq_len, normalise_window=True):
    x_range = range(8000)
    #matrix = [math.sin(x*0.05)*math.sin(x*0.15*x)*math.sin(x*0.35)+2 for x in x_range]
    matrix = [math.sin(x*0.05)+2 for x in x_range]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(matrix) - sequence_length):
        result.append(matrix[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.seed(100)
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
        normalised_window = [(float(p) - float(window[0])) / float(window[0]) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
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
