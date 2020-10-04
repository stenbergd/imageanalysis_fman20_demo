import numpy as np

# Corresponds to Matlab function rgb2gray, see https://www.mathworks.com/help/matlab/ref/rgb2gray.html
def rgb2gray(image):
	return np.dot(image[...,:3], np.array([0.2989, 0.5870, 0.1140]))
