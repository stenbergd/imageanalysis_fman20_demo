# Illustrate deep learning for character recognition
# Example data and deep learnt net from exercise 4 
# in matconvnet practical tutorial
# See: 
# http://github.com/vlfeat/matconvnet/
# and 
# http://www.robots.ox.ac.uk/~vgg/practicals/cnn/
# 
# http://cs.stanford.edu/people/karpathy/convnetjs/

import os
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

## Load the data
mat = loadmat(os.path.join('data', 'ex4data.mat'))
im = mat['im']
res = mat['res']

## Visualize one image
plt.close('all')
plt.figure(1)
plt.imshow(im, cmap='gray')
plt.axis('equal')

## The last layer here gives the response before softmax
# rows are the 26 classes 'a' to 'z'
# colums are positions (x) along the image
blubb = np.squeeze(res[...,7]['x'][0]).T
plt.figure(2)
plt.imshow(blubb, cmap='gray')

## Do softmax on the response.
# rows are the 26 classes 'a' to 'z'
# colums are positions (x) along the image

blubb2 = np.exp(blubb)/np.tile(np.sum(np.exp(blubb),0),(26,1))
plt.figure(3)
plt.imshow(blubb2, cmap='gray')

## Get the classification as the class with the maximal value after softmax
maxi=np.argmax(blubb2, 0)
alfabet = np.array(list('abcdefghijklmnopqrstuvwxyz'))

## This is the ocr result after sweeping across the image

print(alfabet[maxi])

plt.show()
