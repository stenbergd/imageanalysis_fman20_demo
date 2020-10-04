## Edge detection using magnitude of gradient

import os
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import pyplot as plt
from imageanalysistools.plot_edgelets import plot_edgelets
from imageanalysistools.optim_edge import optim_edge
from imageanalysistools.normangle2 import normangle2

im = plt.imread(os.path.join('data', 'klossar.pgm'))
plt.figure(1)
plt.imshow(im, cmap='gray')

## Generate convolution kernels 

a = 2 # level of smoothing, i.e. choice of scale in scale-space.
N = int(np.ceil(a*4)) # Since the exponential degreas fast. Taking 4*scale
			   # is good enough, we hope.
x,y=np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1)) # Generate grid

## First the one representing smoothing + taking derivative w r t x

ddx = -(1/(2*np.pi*a**2)*2*x*np.exp( -(x**2+y**2)/(2*a**2)) )
plt.figure(2)
plt.imshow(ddx, cmap='gray')
plt.title('Kernel representing derivative w r t x + smoothing')

## First the one representing smoothing + taking derivative w r t y

ddy = -(1/(2*np.pi*a**2)*2*y*np.exp( -(x**2+y**2)/(2*a**2)) )
plt.figure(3)
plt.imshow(ddy, cmap='gray')
plt.title('Kernel representing derivative w r t y + smoothing')

## Put these two together in a small bank of filters
# This seems really unneccessary for this simple example
# but might be useful thinking for examples later in the course.

filterbank = np.zeros((2*N+1, 2*N+1, 2))
filterbank[0:(2*N+1),0:(2*N+1),0] = ddx
filterbank[0:(2*N+1),0:(2*N+1),1] = ddy

## Go from image -> result of filterbank

layer2 = np.zeros((np.shape(im)[0],np.shape(im)[1],np.shape(filterbank)[2]))
for k in range(0, np.shape(filterbank)[2]):
	layer2[:,:,k]=convolve2d(im,filterbank[:,:,k],'same')

## Display channel 1 of the layer1, the smoothed d(im)/dx

plt.figure(4)
plt.imshow(layer2[:,:,0], cmap='gray')
plt.title('The derivative of im w r t x + smoothing')

## Display channel 2 of the layer1, the smoothed d(im)/dy

plt.figure(5)
plt.imshow(layer2[:,:,1], cmap='gray')
plt.title('The derivative of im w r t y + smoothing')

## Genarate layer 3, 
# This usually only has one channel, i.e. 
#   the norm of the gradient at each pixel.
# But you might also calculate the angle of the gradient
#   and store this as a second channel.

layer3 = np.zeros((np.shape(im)[0],np.shape(im)[1],np.shape(filterbank)[2]))
layer3[:,:,0] = np.sqrt(layer2[:,:,0]**2 + layer2[:,:,1]**2 )
layer3[:,:,1] = np.arctan2(layer2[:,:,0],layer2[:,:,1])

## The angle of the gradient of smoothed image

plt.figure(7)
plt.imshow(layer3[:,:,1], cmap='hot')
plt.title('The angle of the gradient of smoothed image')
plt.colorbar()

## The norm of the gradient of smoothed image

plt.figure(8)
plt.imshow(layer3[:,:,0], cmap='gray')
plt.title('The norm of the gradient of smoothed image')
plt.axis([140, 160, 150, 130])

## Non-maximum suppression. 
# Classify gradient in 8 sectors - 8 direction bins
# For each bin, keep an edge point if it is local maximum in 
# this direction. 

dirbin = np.mod(np.round(layer3[:,:,1]*8/(2*np.pi)),8)
dx = [0, 1, 1, 1, 0, -1, -1, -1]
dy = [1, 1, 0, -1, -1, -1, 0, 1]
m,n = np.shape(im)
imid = np.arange(1,m-1)
jmid = np.arange(1,m-1)
ok = np.zeros((m,n,8))
for k in range(0, 8):
	ok[1:m-1,1:m-1,k]= (dirbin[np.ix_(imid,jmid)]==k) & \
		(layer3[np.ix_(imid,jmid)][...,0] >= layer3[np.ix_(imid+dy[k],jmid+dx[k])][...,0]) &  \
		(layer3[np.ix_(imid,jmid)][...,0] >= layer3[np.ix_(imid-dy[k],jmid-dx[k])][...,0])
plt.figure(9)
plt.imshow(np.sum(ok,2)*layer3[:,:,0], cmap='gray')
plt.axis([140, 160, 150, 130])

## Plot histogram of log of norm of gradient. 
# Use this to choose threshold

tmp1 = np.sum(ok,2)>0
gra = layer3[:,:,0]
th = layer3[:,:,1]
tmp = np.argwhere( ((tmp1) & (gra > np.exp(-5))).flatten() )
plt.figure(10)
plt.hist(np.log(gra.flatten()[tmp]),np.arange(-5,6.1,0.1))
plt.title('Histogram of log of norm of gradients')


## Combine non-maximum supression with thresholding. 
# result is a discrete set of pixels. 
# The edge is found to pixel precision.

T = np.exp(3)
newok = tmp1 & (gra>T)
plt.figure(11)
plt.imshow(newok*gra, cmap='gray')
plt.axis([140, 160, 150, 130])


## Pixel precision edgelets
# At each edge point we also have an estiamate of the 
# orientation. Each pixel can thus be thought of as 
# a small edge-part, an edgelet. These are shown as
# small red lines. Notice that the orientetion is 
# quite well estimated, wheras the edge is only ok up
# to one pixel. 


y=np.argwhere(newok)[...,0]
x=np.argwhere(newok)[...,1]
tt = np.pi/2-th[newok]
edgelets = np.array([x.T,y.T,tt.T])

plt.figure(12)
plt.clf()
plt.imshow(newok*gra, cmap='gray')
plot_edgelets(edgelets,'r')
plt.axis([140, 160, 150, 130])
plt.title('Each edge point with its direction shown as a small edgelet.')

## Edgelets overlayed on original image.

plt.figure(13)
plt.clf()
plt.imshow(im, cmap='gray')
plot_edgelets(edgelets,'r')
plt.axis([140, 160, 150, 130])
plt.title('Edgelets overlayed on original image.')

## Do sub-pixel refinement.
# Using local optimization and interpolation
# the edgelets can be refined to sub-pixel positions.

edgelets_sub = edgelets.copy()
for k in range(0, np.shape(edgelets)[1]):
	edgelets_sub[:,k],_,_=optim_edge(im,edgelets[:,k])

## Plot edglets before (red) and after (yellow) sub-pixel refinement

plt.figure(14)
plt.clf()
plt.imshow(im, cmap='gray')
plot_edgelets(edgelets,'r')
plot_edgelets(edgelets_sub,'y')
plt.title('Edglets before (red) and after (yellow) sub-pixel refinement')
plt.axis([140, 160, 150, 130])

## Connect edgelets
# Think of each edglelet as nodes/vertices in an graph
# and connect two of them if they are
# * Close to each other
# * have similar orientation
# * the first midpoint lie on the line defined by the second edgelet
# * and vice verca

T1 = 3 # Edgelets have to be within T1 from each other
T2 = 0.1 # Edgelets have to have an orientation difference less than T2 radians
T3 = 0.2 # Mid point must lie closer than T3 pixels from the line defined by
# other edglet.
ab = [np.cos(edgelets_sub[2,:]),np.sin(edgelets_sub[2,:])]
c = -np.sum(ab*edgelets_sub[0:2,:], 0)
l = np.append(ab,[c], 0)
N = np.shape(edgelets_sub)[1]
ph = np.append(edgelets_sub[0:2,:],[np.ones((N))], 0)

I = np.array([])
J = np.array([])
for i in range(0, N):
	jtmp = np.argwhere( np.sqrt(np.sum( (edgelets_sub[0:2,:]-np.tile(np.array([edgelets_sub[0:2,i]]).T,(1,N)))**2, 0) ) < T1 )
	ok2 =  (np.abs( normangle2(edgelets_sub[2,jtmp] - np.tile(edgelets_sub[2,i],(1,len(jtmp)))) ) < T2) \
		& ( np.abs( np.sum(ph[:,jtmp][...,0] * np.tile(np.array([l[:,i]]).T,(1,len(jtmp))), 0) ) < T3 )
	jj = np.setdiff1d(jtmp[np.argwhere(ok2)],i)
	I = np.append(I, np.tile(i,(1,len(jj))))
	J = np.append(J, jj)

## Form graph
G = csr_matrix((np.ones(np.shape(I)), (I, J)), shape=(N, N))

## Find connected component in graph

nrcomp,compind=connected_components(G)

## Plot all connected edgelet components with more than 2 elements
oklong = np.zeros(np.shape(compind))
for k in range(0, nrcomp):
	sel = np.argwhere(compind==k)
	#len(sel)
	if len(sel)>5:
		oklong[sel]=np.ones(np.shape(sel))
		plt.figure(15)
		plt.clf()
		plt.imshow(im, cmap='gray')
		plot_edgelets(edgelets_sub[:,sel],'y')
		plt.title('Edglets of component: ' + str(k) + ' with ' + str(len(sel)) + ' elements.')
		#plt.waitforbuttonpress()

## Plot edgelets of components with length longer than 5 elements

plt.figure(16)
plt.clf()
plt.imshow(im, cmap='gray')
plot_edgelets(edgelets_sub[:,np.argwhere(oklong)],'y')
plt.title('Edglets of components with length longer than 5 elements.')

## Plot edgelets of components with length longer than 5 elements

plt.figure(17)
plt.clf()
plt.imshow(im, cmap='gray')
plot_edgelets(edgelets_sub[:,np.argwhere(oklong)],'y')
plt.title('Edglets of components with length longer than 5 elements.')
plt.axis([130, 170, 160, 120])

plt.show()
