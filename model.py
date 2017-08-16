import os
import time
import numpy as np
from keras.models import Sequential

def build_model():
	model = Sequential()
	
	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print("> Compilation Time : ", time.time() - start)
	return model
