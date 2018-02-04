# Implements multiple CNN branches of different stride+kernel sizes
import os
import numpy as np
from keras.models import Sequential

def build_model(layers, ksize1, ksize2, ksize3):
    dropout = 0.5
    conv_stride = 2
    pool_size = 2
    padding = "same"

    lstm_units = 400
    stride_1 = 3
    filter_num = 128
    cnn_layers = 3

    branch1 = build_cnn_layers(layers, ksize1, stride_1, filter_num, cnn_layers)

    branch2 = build_cnn_layers(layers, ksize2, stride_1, filter_num, cnn_layers)

    branch3 = build_cnn_layers(layers, ksize3, stride_1, filter_num, cnn_layers)

    model = Sequential()
    model.add(Merge([branch1, branch2, branch3], mode='concat'))
    model.add(Dense(
        output_dim = 1))
    model.add(Activation('linear'))

    model.compile(loss="mse", optimizer="adadelta")
    return model;

def build_cnn_layers(layers, lstm_units, kernel_size, stride_1, filter_num, cnn_layers):
    branch = Sequential()

    branch.add(Conv1D(
        input_shape = (layers[1], layers[0]),
        filters = filter_num,
        kernel_size = ksize,
        strides = stride1,
        padding = padding,
        activation = None))
    BatchNormalization(axis=-1)
    branch.add(Activation('relu'))

    branch.add(MaxPooling1D(
        pool_size = pool_size))

    for x in range(1, cnn_layers):
        branch.add(Conv1D(
            filters = filter_num*2*x,
            kernel_size = ksize,
            strides = conv_stride,
            padding = padding,
            activation = None))
        BatchNormalization(axis=-1)
        branch.add(Activation('relu'))
        branch.addMaxPooling1D(
                pool_size = pool_size))

    branch.add(LSTM(
        lstm_units,
        return_sequences = True))
    branch.add(Dropout(dropout/2))

    branch.add(LSTM(
        lstm_units,
        return_sequences = False))
    branch.add(Dropout(dropout))

    return branch
