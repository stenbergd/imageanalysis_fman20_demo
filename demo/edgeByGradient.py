## Edge detection using magnitude of gradient

import os
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

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

plt.figure(6)
plt.imshow(layer3[:,:,0], cmap='gray')
plt.title('The norm of the gradient of smoothed image')

## Summary of the computational steps

# Generate the filterbanks. This could be pre-processed and saved.
a = 2 # level of smoothing, i.e. choice of scale in scale-space.
N = int(np.ceil(a*4)) # Since the exponential degreas fast. Taking 4*scale               # is good enough, we hope.
x,y=np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1)) # Generate grid
ddx = -(1/(2*np.pi*a**2)*2*x*np.exp( -(x**2+y**2)/(2*a**2)) )
ddy = -(1/(2*np.pi*a**2)*2*y*np.exp( -(x**2+y**2)/(2*a**2)) )
filterbank[0:(2*N+1),0:(2*N+1),0] = ddx
filterbank[0:(2*N+1),0:(2*N+1),1] = ddy

# The actual computations
layer2 = np.zeros((np.shape(im)[0],np.shape(im)[1],np.shape(filterbank)[2]))
for k in range (0, np.shape(filterbank)[2]):
	layer2[:,:,k]=convolve2d(im,filterbank[:,:,k],'same')
layer3[:,:,0] = np.sqrt(layer2[:,:,0]**2 + layer2[:,:,1]**2 )
layer3[:,:,1] = np.arctan2(layer2[:,:,0],layer2[:,:,1])

plt.show()
