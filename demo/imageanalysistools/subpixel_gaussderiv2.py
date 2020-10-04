import numpy as np

def subpixel_gaussderiv2(bild,punkt,theta,a):

	x0=punkt[0]
	y0=punkt[1]
	m,n=np.shape(bild)
	NN=np.round(a*3)
	cutx=np.arange(np.max(np.round(x0)-NN,0),min(round(x0)+NN,n)).astype(int)
	cuty=np.arange(np.max(np.round(y0)-NN,0),min(round(y0)+NN,m)).astype(int)
	#
	#[slask,sx]=size(cutx)
	#[slask,sy]=size(cuty)
	#if (sx*sy == 0):  m1=0 m2=0 m3=0

	cutbild=bild[np.ix_(cuty,cutx)]
	x,y=np.meshgrid(cutx,cuty)

	filt1 = (2*(x-x0)*np.cos(theta)+2*(y-y0)*np.sin(theta))/(a**4*np.pi)*np.exp(-((x-x0)**2+(y-y0)**2)/(a**2))
	filt2 = -(2*np.cos(theta)**2+2*np.sin(theta)**2)/(a**4*np.pi)*np.exp(-((x-x0)**2+(y-y0)**2)/(a**2))+ \
			 (-2*(x-x0)*np.cos(theta)-2*(y-y0)*np.sin(theta))**2/(a**6*np.pi)*np.exp(-((x-x0)**2+(y-y0)**2)/(a**2))
	filt3 = 3*(        2*np.cos(theta)**2+2*np.sin(theta)**2)/(a**6*np.pi)* \
			  (-2*(x-x0)*np.cos(theta)-2*(y-y0)*np.sin(theta))*np.exp(-((x-x0)**2+(y-y0)**2)/(a**2)) \
	-(-2*(x-x0)*np.cos(theta)-2*(y-y0)*np.sin(theta))**3/(a**8*np.pi)*np.exp(-((x-x0)**2+(y-y0)**2)/(a**2))
	m1=np.sum(np.sum(filt1*cutbild))
	m2=np.sum(np.sum(filt2*cutbild))
	m3=np.sum(np.sum(filt3*cutbild))

	return m1, m2, m3
