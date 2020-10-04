## Illustration of interpolation in 1D

import numpy as np
from matplotlib import pyplot as plt
from imageanalysistools.myinterpolate1d import myinterpolate1d
from imageanalysistools.h1 import h1
from imageanalysistools.h2 import h2

# Here is a discrete 1D signal

w = [1, 5, 3, 4, 1, 2]
i = np.arange(1,7)
plt.figure(1)
plt.plot(i,w,'*')
plt.axis([-3, 10, 0, 20])

##

## The interpolation expression is W(x) = \sum_i w(i) h(x-i)
# Here h(x) is a small 'bump' that is scaled with w(i) and
# moved to position i.
# Let's illustrate this with a wierd interpolation function
# h(x)

x = np.arange(-5, 5.05, 0.05)
hx = h2(x)
plt.figure(2)
plt.clf()
plt.plot(x,hx,'b','LineWidth',3)
ii = np.arange(-5,6)
deltaii = np.zeros(np.shape(ii))
deltaii[5]=1
plt.plot(ii,deltaii,'r*')


##

interpfun = h2

xx = np.arange(-2,9.01,0.01)
W = myinterpolate1d(w,xx,interpfun)
Ws = [ W ]
x = np.arange(-5, 5.05, 0.05)
hx = interpfun(x)

plt.figure(3)
plt.clf()
plt.subplot(1,2,1)
plt.plot(x,hx,'b','LineWidth',3)
plt.axis([-3, 3, 0, 5])
plt.plot(ii,deltaii,'r*')
plt.subplot(1,2,2)
plt.plot(i,w,'*')
plt.axis([-3, 10, 0, 20])
plt.plot(xx,W,'b-')

##

interpfun = h1

xx = np.arange(-2,9.01,0.01)
W = myinterpolate1d(w,xx,interpfun)
Ws.append(W)
x = np.arange(-5, 5.05, 0.05)
hx = interpfun(x)

plt.figure(4)
plt.clf()
plt.subplot(1,2,1)
plt.plot(x,hx,'b','LineWidth',3)
plt.axis([-3, 3, 0, 5])
plt.plot(ii,deltaii,'r*')
plt.subplot(1,2,2)
plt.plot(i,w,'*')
plt.axis([-3, 10, 0, 20])
plt.plot(xx,W,'b-')

##

interpfun = np.sinc

xx = np.arange(-2,9.01,0.01)
W = myinterpolate1d(w,xx,interpfun)
Ws.append(W)
x = np.arange(-5, 5.05, 0.05)
hx = interpfun(x)

plt.figure(5)
plt.clf()
plt.subplot(1,2,1)
plt.plot(x,hx,'b','LineWidth',3)
plt.axis([-3, 3, 0, 5])
plt.plot(ii,deltaii,'r*')
plt.subplot(1,2,2)
plt.plot(i,w,'*')
plt.axis([-3, 10, 0, 20])
plt.plot(xx,W,'b-')

## What happens after smoothing

mycolors = ['r','g','b']
for aa in np.arange(0.1, 2.01, 0.1):
	xxx = np.arange(-10,10.01,0.01)
	ggg = 1/np.sqrt(2*np.pi*aa**2)*np.exp( - xxx**2/(2*aa**2 ))
	
	plt.figure(6)
	plt.clf()
	plt.subplot(1,2,1)
	for kk in range(0,3):
		plt.plot(x,hx,'b','LineWidth',3)
	
	plt.axis([-3, 3, 0, 5])
	plt.plot(ii,deltaii,'r*')
	plt.subplot(1,2,2)
	plt.plot(i,w,'*')
	plt.axis([-3, 10, 0, 20])
	for kk in range(0,3):
		plt.plot(xx,np.convolve(Ws[kk],ggg,'full')[len(ggg) -1 - (len(ggg)-1)//2:len(ggg)-1 - (len(ggg)-1)//2 + len(Ws[kk])]/100,mycolors[kk],'LineWidth',3)
	plt.title(str(aa))

plt.show()
