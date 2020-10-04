import numpy as np

def psphere(X):
	Xut = np.zeros(np.shape(X))

	slask,n=np.shape(X) if len(np.shape(X)) > 1 else np.shape(X)+(1,)
	if n == 1:
		Xut = X/np.linalg.norm(X)
	else:
		for i in range(0, n):
			Xut[i]=X[i]/np.linalg.norm(X, axis=0)

	return Xut
