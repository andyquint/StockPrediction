import lstm
import cnn_batchnorm_lstm
import time
import matplotlib.pyplot as plt
import dataload
from datetime import datetime
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from keras.callbacks import EarlyStopping
import pandas as pd

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig, axs = plt.subplots(len(predicted_data), 1, sharex=True)
    if (len(predicted_data) > 1):
        for x in range(len(predicted_data)):
            axs[x].plot(true_data, label='True Data')
            # Pad the list of predictions to shift it in the graph to it's correct start
            for i, data in enumerate(predicted_data[x][0]):
                padding = [None for p in range(i * prediction_len)]
                axs[x].plot(padding + data, label='Prediction')
            axs[x].set_title(predicted_data[x][1])
    else:
        axs.plot(true_data, label='True Data')
        for i, data in enumerate(predicted_data[0][0]):
            padding = [None for p in range(i * prediction_len)]
            axs.plot(padding + data, label='Prediction')
        axs.set_title(predicted_data[0][1])
    plt.show()

if __name__=='__main__':
    global_start_time = time.time()
    batch_size = 64
    epochs  = 100
    seq_len = 50
    predict_len = 20

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = dataload.load_data('daily_spx.csv', seq_len, True)

    print('> Data Loaded. Compiling...')

    kernel_sizes = [14]
    early_stopping = EarlyStopping(patience=2)
    fit_params = dict(callbacks=[early_stopping])
    model = KerasRegressor(build_fn=cnn_batchnorm_lstm.build_model, batch_size=batch_size, validation_split = 0.10)
    param_grid=dict(
            layers=[(1, seq_len)], 
            epochs=[epochs], 
            cnn_layers=[3],
            lstm_units=[200,300],
            kernel_size=[15,30],
            stride_1=[3,4]
            )
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[early_stopping])
    print(grid.best_params_)
    print(grid_result)

    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)

    print(results.to_string())

    results.to_csv('results_{}.csv'.format(int(datetime.now().timestamp())))

    exit()
    cnn_models = [
            cnn_batchnorm_lstm.build_model([1, seq_len], 3, ksize) for ksize in kernel_sizes
            ]
    [model.fit(
        X_train,
        y_train,
        batch_size=64,
        nb_epoch=epochs,
        validation_split=0.05,
        callbacks=[early_stopping])
        for model in cnn_models
        ]

    for i in range(len(kernel_sizes)):
        cnn_models[i].save('model-ksize-{}.h5'.format(kernel_sizes[i]))

    #lstm_predictions = dataload.predict_sequences_multiple(lstm_model, X_test, seq_len, seq_len)
    cnn_predictions = [dataload.predict_sequences_multiple(model, X_test, seq_len, predict_len)
            for model in cnn_models]
    #predictions = dataload.predict_sequence_full(model, X_test, seq_len)
    #predictions = dataload.predict_point_by_point(model, X_test)        

    #lstm_score = lstm_model.evaluate(X_test, y_test, verbose=0)
    cnn_scores = [model.evaluate(X_test, y_test, verbose=0)
            for model in cnn_models]

    #print('LSTM')
    #print(lstm_score)
    #print('Test loss: ', lstm_score[0])
    #print('Test accuracy: ', lstm_score[1])
    for i in range(len(kernel_sizes)):
        print('\nKernel size {}'.format(kernel_sizes[i]))
        print(cnn_scores[i])

    print('Training duration (s) : ', time.time() - global_start_time)
    cnn_plots = [(cnn_predictions[i], 'CNN {}'.format(kernel_sizes[i])) for i in range(len(kernel_sizes))]
    plot_results_multiple(cnn_plots, y_test, predict_len)
