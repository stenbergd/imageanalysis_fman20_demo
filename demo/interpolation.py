## Interpolation and low pass smoothing

import os
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from imageanalysistools.rgb2gray import rgb2gray

# Load in an image of gold atoms
# convert to grey-scale and to double precision
# floating point format (with double)

im = plt.imread(os.path.join('data', 'Sint_Au_865200X_19_cut.jpg'))
img = np.round(rgb2gray(im))
img = img[29:50,29:45]

## Plot the original image

plt.figure(1)
plt.imshow(img, cmap='gray')
plt.title('Original discrete image.')

## Think of f as a discrete image
f = img

## Introduce a discrete coordinate system
# For this example we will let i be the integer coordinates
# in the x-direction
# and j be the integer coordinate in the y-direction

i,j=np.meshgrid(np.arange(1,np.shape(f)[1]+1),np.arange(1,np.shape(f)[0]+1))

## interpolate at point (u,v)
# using the equation $F(u,v) = \sum_i \sum_j f(i,j) g(x-i,y-j)$
# where g is the interpolation function
# In Python this is
# F(u,v) = sum(sum( f*g(x-i,y-j) ))
# or F(u,v) = sum(sum(f*h)), with h(i,j) = g(x-i,y-j)

x = 8.1
y = 12.3
sigma = 1.2
h = 1/(2*np.pi*sigma**2) * np.exp( -( (x-i)**2 + (y-j)**2 )/(2*sigma**2) ) # Here: h(i,j) = g(x-i,y-j)
F   = sum(sum(f*h))                                               # Here is the equation
print('The interpolated value F(8.1,12.3) = ' + str(F))

## The interpolated value at coordinates (x,y) = (8.1,12.3)
# is pretty close to the value at (x,y)=(8,12)
# of f convoluted with g

xx,yy=np.meshgrid(np.arange(-5,5),np.arange(-5,5))
gg = 1/(2*np.pi*sigma**2) * np.exp( -( xx**2 + yy**2 )/(2*sigma**2) )

fconv = convolve2d(f,gg,'same')
print(fconv[11,7])

print('The value of F(8,12) = ' + str(F) + ' is pretty close as it should be.')

## Image smoothed with Gaussian of width sigma = 1.2

plt.figure(2)
plt.imshow(fconv, cmap='gray')
plt.title('Image smoothed with Gaussian of width sigma = 1.2')

plt.show()
