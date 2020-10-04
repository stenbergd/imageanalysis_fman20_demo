import numpy as np

def myinterpolate1d(w,xlist,interpfun):
	# W = myinterpolate1d(w,xlist,interpfun)

	i = np.arange(1,np.size(w)+1)
	W = np.zeros(np.shape(xlist))
	for k in range(0, np.size(xlist)):
		x = xlist[k]
		W[k] = sum(interpfun(x-i)*w)
		# W(x) = sum_i w(i) h(x-i)

	return W
