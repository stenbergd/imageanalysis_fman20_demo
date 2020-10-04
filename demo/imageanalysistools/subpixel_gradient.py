import numpy as np

def subpixel_gradient(bild,punkt,a):
# [matx,maty,komplext]=subpixel_gradient(bild,punkt,a);
# Input
#   bild  - input image
#   punkt - point
#   a     - amount of smoothing, should be greater than 1 (kind of)
# Output
#   matx - x-component of gradient
#   maty - y-component of gradient
#   komplext
#   th   - angle
	x0=punkt[0]
	y0=punkt[1]
	m,n=np.shape(bild)
	NN=np.round(a*3)
	cutx=np.arange(np.max(np.round(x0)-NN,0),min(round(x0)+NN,n)).astype(int)
	cuty=np.arange(np.max(np.round(y0)-NN,0),min(round(y0)+NN,m)).astype(int)
	cutbild=bild[np.ix_(cuty,cutx)]
	x,y=np.meshgrid(cutx,cuty)
	filtx=2*(x-x0)*np.exp(- ((x-x0)**2 + (y-y0)**2)/a**2)/(a**4*np.pi)
	filty=2*(y-y0)*np.exp(- ((x-x0)**2 + (y-y0)**2)/a**2)/(a**4*np.pi)
	matx=np.sum(np.sum(filtx*cutbild))
	maty=np.sum(np.sum(filty*cutbild))
	komplext=matx+np.sqrt(-1+0j)*maty
	th = np.arctan2(maty,matx)

	return matx,maty,komplext,th

