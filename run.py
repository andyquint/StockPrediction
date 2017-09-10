import lstm
import time

if __name__=='__main__':
	global_start_time = time.time()
	#epochs  = 1
	epochs  = 100
	seq_len = 50

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = lstm.load_data('goog.csv', seq_len, True)
	#X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)

	print('> Data Loaded. Compiling...')

	model = lstm.build_model([1, 50, 100, 1])

	model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)

	predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	#predicted = lstm.predict_point_by_point(model, X_test)        

	print('Training duration (s) : ', time.time() - global_start_time)
