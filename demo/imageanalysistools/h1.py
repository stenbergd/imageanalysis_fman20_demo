import numpy as np

def h1(x):

	h = np.zeros(np.shape(x))
	i = np.argwhere( (x>-0.8) & (x<=-0.5) )
	h[i] = (40/9)*(x[i]+0.8)
	i = np.argwhere( (x>-0.5) & (x<=-0.2) )
	h[i] = -(40/9)*(x[i]+0.2)

	i = np.argwhere( (x>-0.2) & (x<=0) )
	h[i] = 5*(x[i]+0.2)
	i = np.argwhere( (x>0) & (x<=0.2) )
	h[i] = -5*(x[i]-0.2)

	i = np.argwhere( (x>0.2) & (x<=0.5) )
	h[i] = (40/9)*(x[i]-0.2)
	i = np.argwhere( (x>0.5) & (x<=0.8) )
	h[i] = -(40/9)*(x[i]-0.8)

	return h
