## Demo of histogram equalization

import os
import numpy as np
from matplotlib import pyplot as plt
from imageanalysistools.rgb2gray import rgb2gray

## Read in an image

inbild = rgb2gray(plt.imread(os.path.join('data', 'mkalle.gif')))
lag = 0
hog = 255

## Round to the nearest integer
# Make sure that no intensities are outside the range
bild=np.round(inbild)
bild=np.clip(bild, lag, hog)
# Transform the range from [lag,hog] to [1,M]
bild=bild+np.ones(np.shape(bild))*(1-lag)
M=hog-lag+1

## Calculate the histogram
hist1,nn1=np.histogram(bild[:],np.arange(1,M+2))

# Calculate the cumulative histogram
chist1=np.cumsum(hist1)
# chist1 is in principle the gray-scale transformation 
# we are going to use. We just have to rescale it
hh=M*chist1/chist1[M-1]
# and round to nearest integer
transf=np.ceil(hh)

## OK. Now we will perform the gray-scale Transformation
utbild=np.zeros(np.shape(bild))
utbild=transf[bild.astype(int)]

## Plot the results
plt.subplot(2,3,1)
plt.plot(np.arange(1,M+1),np.arange(1,M+1))
plt.axis([0, M, 0, M])
plt.title('The original gray-scales')

plt.subplot(2,3,2)
plt.imshow(bild, cmap='gray')
plt.title('gives this image')

plt.subplot(2,3,3)
plt.plot(chist1)
plt.axis([0, M, 0, max(chist1)])
plt.title('and this cumulative histogram')

plt.subplot(2,3,4)
plt.plot(transf)
plt.axis([0, M, 0, M])
plt.title('This gray-scale transformation')

plt.subplot(2,3,5)
plt.imshow(utbild, cmap='gray')
plt.title('gives this image')

hist2,nn2=np.histogram(utbild[:],np.arange(1,M+1))
chist2=np.cumsum(hist2)
plt.subplot(2,3,6)
plt.plot(chist2)
plt.axis([0, M, 0, max(chist2)])
plt.title('and this cumulative histogram.')

plt.show()
