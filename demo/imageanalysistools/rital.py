import numpy as np
from matplotlib import pyplot as plt
from .psphere import psphere
from .pflat import pflat

def rital(linjer,st='-'):
	if np.shape(linjer)==0:
		slask=[]
	else:
		slask,nn=np.shape(linjer) if len(np.shape(linjer)) > 1 else np.shape(linjer)+(1,)
		rikt=psphere(np.append(np.array([linjer[1],-linjer[0]]),np.zeros((1,nn))))
		punkter=pflat(np.cross(rikt,linjer))
		if nn == 1:
			plt.plot([punkter[0]-4000*rikt[0], punkter[0]+4000*rikt[0]], \
						[punkter[1]-4000*rikt[1], punkter[1]+4000*rikt[1]],st)
		else:
			for i in range(0, nn):
				plt.plot([punkter[0,i]-4000*rikt[0,i], punkter[0,i]+4000*rikt[0,i]], \
						[punkter[1,i]-4000*rikt[1,i], punkter[1,i]+4000*rikt[1,i]],st)
		slask=[]

	return slask
