import numpy as np
from matplotlib import pyplot as plt

def plot_edgelets(xyth,plotst):
#
	for k in range(np.shape(xyth)[1]):
		x = xyth[0][k]
		y = xyth[1][k]
		th = xyth[2][k]
		
		n = [np.cos(th),np.sin(th)]
		t = [-np.sin(th),np.cos(th)]
		plt.plot([x-0.45*t[0], x+0.45*t[0]],[y-0.45*t[1], y+0.45*t[1]],plotst);
		#plt.plot(x,y,'r.')
