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

def plot_results_multiple(predicted_data, true_data, prediction_len, fig_path=None):
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
    if (fig_path is not None):
        fig.savefig(fig_path)
    plt.show()

if __name__=='__main__':
    global_start_time = time.time()
    batch_size = 64
    epochs  = 100
    seq_len = 50
    predict_len = 20

    print('> Loading data... ')

    #X_train, y_train, X_test, y_test = dataload.load_data('daily_spx.csv', seq_len, True)
    X_train, y_train, X_test, y_test = dataload.load_sin_data(seq_len)

    print('> Data Loaded. Compiling...')

    # Grid search parameters
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

    # Grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[early_stopping])
    print('Best parameters:')
    print(grid.best_params_)

    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)

    print(results.to_string())

    # Build top 3 models
    idx = 0
    top_models = []
    for index, row in results.iterrows():
        top_models.append(
                cnn_batchnorm_lstm.build_model(row['param_layers'], row['param_cnn_layers'], row['param_kernel_size'])
            )
        idx = idx + 1
        if idx > 2:
            break

    for model in top_models:
        model.fit(
                X_train,
                y_train,
                batch_size=64,
                nb_epoch=epochs,
                validation_split=0.05,
                callbacks=[early_stopping]
                )

    predictions = [dataload.predict_sequences_multiple(model, X_test, seq_len, predict_len)
            for model in top_models]
    scores = [model.evaluate(X_test, y_test, verbose=0)
            for model in top_models]

    # Save results
    results_fname = 'results_{}'.format(int(datetime.now().timestamp()))
    results.to_csv('{}.csv'.format(results_fname))
    top_model_plots = [(predictions[i], 'Model {}'.format(i+1)) for i in range(len(predictions))]
    plot_results_multiple(top_model_plots, y_test, predict_len, fig_path = '{}.pdf'.format(results_fname))
