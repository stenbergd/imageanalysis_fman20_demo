import numpy as np

def normangle2(angle):
# n=normangle2(angle)
# calculates angle modolu pi. 
# normalizes angle in the interval [-pi/2,pi/2].

	return np.remainder(angle+8*np.pi+np.pi/2,np.pi)-np.pi/2