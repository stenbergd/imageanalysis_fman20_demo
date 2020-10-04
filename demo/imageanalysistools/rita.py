import numpy as np
from matplotlib import pyplot as plt

def rita(bild,st='-'):
	if np.shape(bild)==0:
		slask=[]
	else:
		plt.plot(bild[0,:],bild[1,:],st)
		slask=[]

	return slask
