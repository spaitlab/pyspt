import scipy.signal

	def butter(n, Wn, btype='low', analog=False, 	output='ba', fs=None):
    	return scipy.signal.butter(n, Wn, btype=btype, analog=analog, output=output, fs=fs)