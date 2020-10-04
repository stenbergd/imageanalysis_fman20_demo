import numpy as np

def h2(x):

	h = np.zeros(np.shape(x))
	i = np.argwhere( (x>-1) & (x<=0) )
	h[i] = 1*(x[i]+1)
	i = np.argwhere( (x>0) & (x<=1) )
	h[i] = -1*(x[i]-1)

	return h
