import lstm
import cnn_batchnorm_lstm
import time
import matplotlib.pyplot as plt
import dataload

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
	#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	fig, axs = plt.subplots(len(predicted_data), 1, sharex=True)
	for x in range(len(predicted_data)):
		axs[x].plot(true_data, label='True Data')
		# Pad the list of predictions to shift it in the graph to it's correct start
		for i, data in enumerate(predicted_data[x][0]):
			padding = [None for p in range(i * prediction_len)]
			axs[x].plot(padding + data, label='Prediction')
			#plt.legend()
		axs[x].set_title(predicted_data[x][1])
	#ax2.plot(true_data, label='True Data')
	#for i, data in enumerate(predicted_data[1][0]):
		#padding = [None for p in range(i * prediction_len)]
		#ax2.plot(padding + data, label='Prediction')
		##plt.legend()
	#ax2.set_title(predicted_data[1][1])
	plt.show()

if __name__=='__main__':
	global_start_time = time.time()
	#epochs  = 1
	epochs  = 100
	seq_len = 40

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = dataload.load_data('daily_spx.csv', seq_len, True)

	print('> Data Loaded. Compiling...')

	#lstm_model = lstm.build_model([1, seq_len, 100, 1])
	model_layers = [1, 2, 3, 4]
	cnn_models = [
			cnn_batchnorm_lstm.build_model([1, seq_len, 100, 1], x) for x in model_layers
		]
	[model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
		for model in cnn_models
	]

	#lstm_predictions = dataload.predict_sequences_multiple(lstm_model, X_test, seq_len, seq_len)
	cnn_predictions = [dataload.predict_sequences_multiple(model, X_test, seq_len, seq_len)
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
	for l in model_layers:
		print('\nCNN ', l)
		print(cnn_scores[l-1])
	#print('Test loss: ', cnn_score[0])
	#print('Test accuracy: ', cnn_score[1])

	print('Training duration (s) : ', time.time() - global_start_time)
	cnn_plots = [(cnn_predictions[l-1], 'CNN {}'.format(l)) for l in model_layers]
	plot_results_multiple(cnn_plots, y_test, seq_len)
	#plot_results(predictions, y_test)
