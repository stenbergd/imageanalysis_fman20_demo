## Load the estimated mean image O and two basis images E1 and E2

import os
import time
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

# Think of this as the new origin O
# and two 'vectors' spanning a two-dimensional subspace of images
# P = O + x1 E1 + x2 E2
# with two coordinates (x1,x2)
# These have been estimated using principal component analysis from
# a number of images.
# O is really the mean m
# and E1 and E2 are the two vectors corresponding to the two
# largest eigenvalues of the covariance matrix, (but reshaped as
# images)

mat = loadmat(os.path.join('data', 'bildbas.mat'))
O = mat['O']
E1 = mat['E1']
E2 = mat['E2']
s1 = mat['s1']
s2 = mat['s2']


## Example 1 - Let (x1,x2) = (s1*cos(th),s2*sin(th)),

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(O, cmap='gray', origin='lower')
plt.title('mean image O')
plt.subplot(1,3,2)
plt.imshow(E1, cmap='gray', origin='lower')
plt.title('Eigen-image E1')
plt.subplot(1,3,3)
plt.imshow(E2, cmap='gray', origin='lower')
plt.title('Eigen-image E2')


## Example 1 - Let (x1,x2) = (s1*cos(th),s2*sin(th)),
# i.e. we move along an ellipse in the x1,x2 plane and
# look at the synthesized images
# P(th) = O + x1(th) E1 + x2(th) E2

for th in np.arange(0,2*np.pi,0.1):
	x = [s1*np.cos(th),s2*np.sin(th)] # Coordinates
	P = O + x[0]*E1 + x[1]*E2
	plt.figure(2)
	plt.imshow(P, cmap='gray', origin='lower')
	plt.title('Theta: ' + str(round(th,2)))
	plt.pause(0.1)

## Example 2 - Let (x1,x2) = (-0.7,0.8)
# look at the synthesized images
# P = O + x1 E1 + x2 E2

plt.figure(3)
plt.subplot(1,2,1)
plt.axis([-1, 1, -1, 1])
x1 = -0.7
x2 = 0.8
plt.plot(x1,x2,'*')
plt.title('(x1,x2) in R^2')
x = [s1*x1,s2*x2] # Coordinates
P = O + x[0]*E1 + x[1]*E2
plt.subplot(1,2,2)
plt.imshow(P, cmap='gray', origin='lower')
plt.title(' P = O + x1 E1 + x2 E2 in R^{3321}')

## Example 3 - Let (x1,x2) = (0.9,0.8)
# look at the synthesized images
# P = O + x1 E1 + x2 E2

plt.figure(4)
plt.subplot(1,2,1)
plt.axis([-1, 1, -1, 1])
x1 = 0.9
x2 = 0.8
plt.plot(x1,x2,'*')
plt.title('(x1,x2) in R^2')
x = [s1*x1,s2*x2] # Coordinates
P = O + x[0]*E1 + x[1]*E2
plt.subplot(1,2,2)
plt.imshow(P, cmap='gray', origin='lower')
plt.title(' P = O + x1 E1 + x2 E2 in R^{3321}')

## Example 4 - Let (x1,x2) = (0.2,-0.7)
# look at the synthesized images
# P = O + x1 E1 + x2 E2

plt.figure(5)
plt.subplot(1,2,1)
plt.axis([-1, 1, -1, 1])
x1 = 0.2
x2 = -0.7
plt.plot(x1,x2,'*')
plt.title('(x1,x2) in R^2')
x = [s1*x1,s2*x2] # Coordinates
P = O + x[0]*E1 + x[1]*E2
plt.subplot(1,2,2)
plt.imshow(P, cmap='gray', origin='lower')
plt.title(' P = O + x1 E1 + x2 E2 in R^{3321}')

## Example 5 - Let (x1,x2) = (-0.2,-0.7)
# look at the synthesized images
# P = O + x1 E1 + x2 E2

plt.figure(6)
plt.subplot(1,2,1)
plt.axis([-1, 1, -1, 1])
x1 = -0.2
x2 = -0.7
plt.plot(x1,x2,'*')
plt.title('(x1,x2) in R^2')
x = [s1*x1,s2*x2] # Coordinates
P = O + x[0]*E1 + x[1]*E2
plt.subplot(1,2,2)
plt.imshow(P, cmap='gray', origin='lower')
plt.title(' P = O + x1 E1 + x2 E2 in R^{3321}')

plt.show()
