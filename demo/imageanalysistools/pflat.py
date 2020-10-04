import numpy as np

def pflat(X):
	Xut = np.zeros(np.shape(X))

	Xut[0]=X[0]/X[0]
	Xut[1]=X[1]/X[1]
	Xut[2]=X[2]/X[2]

	return Xut
