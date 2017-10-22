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

	lstm_model = lstm.build_model([1, seq_len, 100, 1])
	cnn1_model = cnn_batchnorm_lstm.build_model([1, seq_len, 100, 1], 1)
	cnn2_model = cnn_batchnorm_lstm.build_model([1, seq_len, 100, 1], 2)
	cnn3_model = cnn_batchnorm_lstm.build_model([1, seq_len, 100, 1], 3)

	lstm_model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
	cnn1_model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
	cnn2_model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
	cnn3_model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)

	lstm_predictions = dataload.predict_sequences_multiple(lstm_model, X_test, seq_len, seq_len)
	cnn1_predictions = dataload.predict_sequences_multiple(cnn1_model, X_test, seq_len, seq_len)
	cnn2_predictions = dataload.predict_sequences_multiple(cnn2_model, X_test, seq_len, seq_len)
	cnn3_predictions = dataload.predict_sequences_multiple(cnn3_model, X_test, seq_len, seq_len)
	#predictions = dataload.predict_sequence_full(model, X_test, seq_len)
	#predictions = dataload.predict_point_by_point(model, X_test)        

	lstm_score = lstm_model.evaluate(X_test, y_test, verbose=0)
	cnn1_score = lstm_model.evaluate(X_test, y_test, verbose=0)
	cnn2_score = lstm_model.evaluate(X_test, y_test, verbose=0)
	cnn3_score = lstm_model.evaluate(X_test, y_test, verbose=0)

	print('LSTM')
	print(lstm_score)
	#print('Test loss: ', lstm_score[0])
	#print('Test accuracy: ', lstm_score[1])
	print('\nCNN 1')
	print(cnn1_score)
	print('\nCNN 2')
	print(cnn2_score)
	print('\nCNN 3')
	print(cnn3_score)
	#print('Test loss: ', cnn_score[0])
	#print('Test accuracy: ', cnn_score[1])

	print('Training duration (s) : ', time.time() - global_start_time)
	plot_results_multiple([(lstm_predictions, 'LSTM'), (cnn1_predictions, 'CNN 1'), (cnn2_predictions, 'CNN 2'), (cnn3_predictions, 'CNN 3')], y_test, seq_len)
	#plot_results(predictions, y_test)
