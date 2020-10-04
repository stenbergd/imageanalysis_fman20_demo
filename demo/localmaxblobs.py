## Here is an image of gold atoms

import os
from matplotlib import pyplot as plt
from imageanalysistools.rgb2gray import rgb2gray

im = double(rgb2gray(plt.imread(os.path.join('data', 'Sint_Au_865200X_19_cut.jpg')))
#im = (plt.imread(os.path.join('data', 'Sint_Au_865200X_19_cut.jpg')))
#im = double(plt.imread('Sint_Au_865200X_19.tif'))
#im = im(1000:1100,1000:1100)
plt.figure(1)
plt.imshow(im, cmap='gray')

## Detection of blobs by local maxima after smoothing with gaussian
# The centre of the gold atoms could be found 
# by finding local maxima in a smothed image
# To low threshold gives some false detections

a =  1  # smoothing factor 4 pixels
th = 0  # threshold 50 for height in pixels
points,out=detect_gaussian_pixel(im,a,th)
plt.figure(2)
plt.imshow(im, cmap='gray')
plt.plot(points[0,:],points[1,:],'*')

## Detection of blobs by local maxima after smoothing with gaussian
# The centre of the gold atoms could be found 
# by finding local maxima in a smothed image
# More smoothing (a=4, coarser scale) 
# combined with higher threshold (th=50) 
# gives fewer false detections.

a =   4 # smoothing factor 4 pixels
th = 50 # threshold 50 for height in pixels
points,out=detect_gaussian_pixel(im,a,th)
plt.figure(3)
plt.imshow(im, cmap='gray')
plt.plot(points[0,:],points[1,:],'*')

## Sub-pixel refinements of blob-peaks

spoints,ok = improve_gaussian_subpixel(im,points)

plt.figure(4)
plt.imshow(im, cmap='gray')
#plt.axis([1000, 1100, 1000, 1100])
plt.plot(points[0,:],points[1,:],'*')
plt.plot(spoints[0,:],spoints[1,:],'ro')

plt.show()
