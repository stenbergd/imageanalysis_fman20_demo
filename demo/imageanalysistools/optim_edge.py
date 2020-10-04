import numpy as np
from .subpixel_gaussderiv2 import subpixel_gaussderiv2
from .subpixel_gradient import subpixel_gradient

def optim_edge(im,xin):
# xin has format [x0,y0,th]

	x0=xin[0]
	y0=xin[1]
	th00 = xin[2]
	#th=pi/2-th00
	th = th00
	xx0 = np.array([x0,y0,th])

	p0 = np.array([x0,y0])
	normal = np.array([np.cos(th),np.sin(th)])
	s = 0
	a = 2
	m10,m2,m3=subpixel_gaussderiv2(im,p0,th,a)
	for j in range(0, 10):
		p=np.array(p0+s*normal)
		#rita(p,'*')
		#     if punkts(2) < 0: keyboard
		# OBS INGEN KONTROLL PA ATT M2/M3 BLIR STOR!!!
		m1,m2,m3=subpixel_gaussderiv2(im,p,th,a)
		s=s-max(min(m2/m3,0.3),-0.3)
		#[m1,m2,m3,s]
	if np.abs(m1)>np.abs(m10):
		# This is good, the norm of the gradient increased! Yeah!
		# Re-estimate the direction of the gradient
		matx,maty,komplext,th=subpixel_gradient(im,p,a)
		xout = np.append(p, th)
		ok = 1
	else:
		xout = np.append(p0, th)
		ok = 0
	res = m2

	return xout, ok, res
