## Getting started with images

import os
import numpy as np
from matplotlib import pyplot as plt
from imageanalysistools.rgb2gray import rgb2gray

# The function imread can be used to read images in e g jpg format to
# matlab. Several other formats are also supported. Write 'help(plt.imread)' for
# more information

im = plt.imread(os.path.join('data', 'dsc_0260.jpg'))

## 
# It is common that images are in colour and that each colour intensity
# is coded as a number between 0 and 255 (8 bits). 

print(np.shape(im))

# Notice that the size is 4816 rows and 2880 columns and three
# colour channels (red, green blue)
# Lets look at the upper left 3x5 pixels. 

print(im[0:3,0:5,:])

# Notice that the pixels have less intensity in the red channel, 
# medium intensity in the green channel and quite a lot of intensity
# in the blue channel. 

## Viewing images
# Images can be viewed using the plt.imshow command. 

plt.figure(1)
plt.imshow(im[0:3,0:5,:])
plt.title('It looks blue doesn\'t it')

##
# Let's look at the whole image

plt.figure(2)
plt.imshow(im)
plt.title('The whole image')

## Image coordinate system
# Image coordinate systems can be confusing. 
# The standard coordinate system for images is to have the
# (1,1) pixel in the upper left corner. 
# This is similar to matrices which are usually written so that
# the first element of the first row is in the upper left corner.

A = np.array([[1, 2, 3],[4, 5, 6],[2, 9, 2]])

print(A[0,0]) # Python uses zero-indexed arrays

# The (1,1) element of A has value 1. When we print the matrix
# we usually put the (1,1) element in the upper left corner. 

print(A)

# It is possible to change from this matrix convention 'ij'
# to more standard 'xy'-coordinate system convention using
# origin='lower' imshow parameter
# origin='higher' imshow parameter

plt.figure(3)
plt.imshow(im, origin='lower')
plt.title('The whole image in xy coordinate convention')

## 
# Another common cause of misunderstanding is that the first index
# of a matrix is used to denote the row.
# But the row is usually considered to be the y-coordinate, which 
# usually is considered to be the 'second' coordinate. 

## Converting to gray-scale
# In the image analysis course we will often study gray-scale images
# It is possible to convert from rgb to grayscale using the
# function rgb2gray (see above for simple Python implementation)
# grayscale images can also be viewed using 'imshow'
# functions. The function 'imshow' first calculates tha lowest ('darkest')
# value and the highest ('brightest') value and then scales the
# image before displaying it. 
# The standard color representation in Python is, however, 
# a pseudo-colour scale from low ('cold') values in blue to
# high ('varm') values in red. 

img = np.round(rgb2gray(im))

# The size is now 4816 x 2880

print(np.shape(img))
print(img[0:3,0:5])
plt.figure(4)
plt.imshow(img)
plt.colorbar()

## colour-maps 
# In the image analysis course it will many times be more appropriate
# to choose a grey-scale colour-map from dark to bright.

plt.figure(5)
plt.imshow(img, cmap='gray')
plt.colorbar()

plt.show()
