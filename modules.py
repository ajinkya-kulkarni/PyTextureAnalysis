#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

########################################################################################

# This file contains all the modules/functions necessary for running the streamlit application or the example notebooks.

########################################################################################

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import cv2

import scipy.ndimage
from scipy import ndimage
from skimage.filters import threshold_mean

import matplotlib.pyplot as plt

########################################################################################

def binarize_image(image):
	"""
	This function checks if the input image is 2D and returns a binary image based on a threshold value.

	Parameters:
	image (np.array): Input image to be thresholded.
	threshold_mean: A method from skimage that calculates the mean threshold value for the input image.

	Returns:
	np.array: Binary image with values above the threshold set to 1 and values below set to 0.

	Raises:
	ValueError: If the input image is not 2D.
	"""
	# Check if the input image is 2D
	if len(image.shape) != 2:
		raise ValueError("Input should be a 2D image.")

	threshold_value = threshold_mean(image)

	binary_image = image > threshold_value

	return binary_image

########################################################################################

def split_into_chunks(img, chunk_size):
	"""
	Splits a 2D grayscale image into chunks of a given size.

	Parameters:
		img (numpy.ndarray): The input 2D grayscale image.
		chunk_size (int): The size of the chunks to split
		the image into.
		overlap_pixels (int): The overlap between chunks,
		in pixels.

	Returns:
		list: A list of chunks, each of
		size chunk_size x chunk_size.
		numpy.ndarray: The padded image, with
		size padded_size x padded_size, where padded_size is a multiple of chunk_size.
	"""
	# Divide the image into chunks
	chunks = []
	for i in range(0, img.shape[0] - chunk_size + 1, chunk_size):
		for j in range(0, img.shape[1] - chunk_size + 1, chunk_size ):
			chunk = img[i:i + chunk_size, j:j + chunk_size]
			chunks.append(chunk)

	return chunks

########################################################################################

def circular_variance(angles):
	"""
	Calculates circular variance of given angles

	Parameters:
	angles (numpy.ndarray): 1D array of angles in radians

	Returns:
	float: circular variance value
	"""
	# check if angles is a 1D array
	if len(angles.shape) != 1:
		raise ValueError("Input must be a 1D array of angles in radians")
	#remove NaN values
	angles = angles[~np.isnan(angles)]
	length = angles.size
	# check if the input has at least one valid value 
	if length == 0:
		raise ValueError("Input must contain at least one valid value")
	# calculate circular variance
	S = np.sum(np.sin(angles))
	C = np.sum(np.cos(angles))
	R = np.sqrt(S**2 + C**2)
	R_avg = R/length
	V = 1 - R_avg
	return V

########################################################################################

def make_coherence(input_image, eigenvalues, Structure_Tensor, threshold_value):
	"""
	Calculate coherence values for a given input image, eigenvalues, structure tensor, and threshold value.

	Parameters:
	- input_image (numpy.ndarray): The input image for which coherence values are to be calculated.
	- eigenvalues (numpy.ndarray): The eigenvalues of the input image.
	- Structure_Tensor (numpy.ndarray): The structure tensor of the input image.
	- threshold_value (float): The threshold value to determine if the calculation should be done.

	Returns:
	- Coherence_Array (numpy.ndarray): An array containing the coherence values for the input image.
	"""
	
	Coherence_Array = np.full(input_image.shape, np.nan)

	# Check if the sum of the EigenValues of the Structure_Tensor is greater than 0
	mask = (input_image >= threshold_value) & (eigenvalues.sum(axis=2) > 0)

	trace = np.trace(Structure_Tensor, axis1=2, axis2=3)
	Smallest_Normalized_Eigenvalues = eigenvalues[:, :, 0] / trace
	Largest_Normalized_Eigenvalues = eigenvalues[:, :, 1] / trace

	# Compute the coherence values using np.where
	Coherence_Array = np.where(mask, np.abs((Largest_Normalized_Eigenvalues - Smallest_Normalized_Eigenvalues) / (Smallest_Normalized_Eigenvalues + Largest_Normalized_Eigenvalues)), Coherence_Array)

	return Coherence_Array

########################################################################################

def convolve(image, kernel):
	"""
	Perform convolution on a binary image with a kernel of any size

	Parameters:
		image (np.ndarray): binary image to perform convolution on
		kernel (np.ndarray): kernel of any size

	Returns:
		np.ndarray: binary image after convolution
	"""

	# Convert the input image to a valid data type in OpenCV
	image = np.array(image, dtype=np.float32)

	# Get the shape of the image
	i_h, i_w = image.shape

	# Get the shape of the kernel
	k_h, k_w = kernel.shape

	# Check if the kernel is of odd size
	if k_h % 2 == 0 or k_w % 2 == 0:
		raise ValueError("Kernel must be of odd size")

	# Check if the kernel size is smaller than the image dimensions
	if k_h > i_h or k_w > i_w:
		raise ValueError("Kernel size must be smaller than image dimensions")

	# Pad the image with the pixels along the edges
	pad_h = int((k_h - 1) / 2)
	pad_w = int((k_w - 1) / 2)
	image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, None, 0)

	# Get the total number of elements in the kernel
	total_elements = k_h * k_w

	# Perform convolution
	result = cv2.filter2D(image, -1, kernel / total_elements, borderType=cv2.BORDER_CONSTANT)

	return result[pad_h:-pad_h, pad_w:-pad_w]

########################################################################################

def make_image_gradients(image, filter=None):
	"""
	Calculates image gradients in x and y directions using filters chosen by the user.

	Parameters:
	image (numpy.ndarray): 2D array representing the input image.
	filter (str): Filter type to be used, either 'sobel' or 'prewitt'

	Returns:
	tuple: Tuple containing 2D arrays representing gradient of the image in x and y directions respectively.

	"""
	# check if input_image is 2D array
	if len(image.shape) != 2:
		raise ValueError("Input image must be a 2D array")

	if filter == None or filter == 'sobel':
		image_gradient_x = ndimage.sobel(image, axis=0)
		image_gradient_y = ndimage.sobel(image, axis=1)

	elif filter == 'prewitt':
		image_gradient_x = ndimage.prewitt(image, axis=0)
		image_gradient_y = ndimage.prewitt(image, axis=1)
	else:
		raise ValueError("Invalid filter type, choose either 'sobel' or 'prewitt'")

	return image_gradient_x, image_gradient_y

########################################################################################

def show_mosaic(chunks, cmap = 'viridis'):
	"""
	Shows a mosaic of the original image, constructed from its chunks, as n x n subplots with reduced borders.

	Parameters:
		img (numpy.ndarray): The input 2D grayscale image.
		chunks (list): A list of chunks, each of size chunk_size x chunk_size.
		overlap_pixels (int): The overlap between chunks, in pixels.
	"""

	# Calculate the number of rows and columns
	n = int(np.ceil(np.sqrt(len(chunks))))

	fig, axs = plt.subplots(n, n, figsize=(8, 8))
	plt.subplots_adjust(wspace=0.05, hspace=0.05)
	for i, chunk in enumerate(chunks):
		row = i // n
		col = i % n
		axs[row, col].imshow(chunk, cmap = cmap)
		axs[row, col].set_xticks([])
		axs[row, col].set_yticks([])

	plt.show()

########################################################################################

def make_orientation(input_image, Jxx, Jxy, Jyy, threshold_value):
	"""
	Calculates orientation of an image using structure tensor components.

	Parameters:
	input_image (numpy.ndarray): 2D array representing the input image.
	Jxx (numpy.ndarray): 2D array representing the xx component of the structure tensor.
	Jxy (numpy.ndarray): 2D array representing the xy component of the structure tensor.
	Jyy (numpy.ndarray): 2D array representing the yy component of the structure tensor.
	threshold_value (float): threshold value for the input image. Only pixels with intensity greater than or equal to threshold_value will be considered.

	Returns:
	numpy.ndarray: 2D array representing the orientation of the input image. Pixels that do not meet the threshold condition are set to NaN.

	"""
	# check if input_image is 2D array
	if len(input_image.shape) != 2:
		raise ValueError("Input image must be a 2D array")
	# check if Jxx, Jxy, Jyy has the same shape as input_image
	if Jxx.shape != input_image.shape or Jxy.shape != input_image.shape or Jyy.shape != input_image.shape:
		raise ValueError("Jxx, Jxy and Jyy must have the same shape as input image")
	# check if threshold_value is a number
	if not isinstance(threshold_value, (float, int)):
		raise ValueError("Threshold value must be a number")

	Orientation = 0.5 * ( cv2.phase( (Jyy - Jxx), (2 * Jxy), angleInDegrees = True) )

	Orientation[input_image < threshold_value] = np.nan

	return Orientation

########################################################################################

def generate_padded_image(img, chunk_size):
	"""
	Generate a padded image that is square and a multiple of the chunk size.

	Parameters:
		img (numpy.ndarray): The input image.
		chunk_size (int): The size of the chunks used for analyzing the image.

	Returns:
		numpy.ndarray: The padded image.
	"""

	# Pad the image to make it square and a multiple of chunk_size
	max_size = max(img.shape)
	padded_size = max_size + (chunk_size - max_size % chunk_size) % chunk_size

	padded_img = np.zeros((padded_size, padded_size))
	padded_img[:img.shape[0], :img.shape[1]] = img

	return padded_img

########################################################################################

def make_structure_tensor_2d(image_gradient_x, image_gradient_y, local_sigma):
	"""
	Calculates 2D structure tensor of an image using image gradients in x and y directions and a local standard deviation.

	Parameters:
	image_gradient_x (numpy.ndarray): 2D array representing the gradient of the image in x direction.
	image_gradient_y (numpy.ndarray): 2D array representing the gradient of the image in y direction.
	local_sigma (float): standard deviation for the Gaussian filter used for calculating the structure tensor.

	Returns:
	tuple: Tuple containing the 2D structure tensor, eigenvalues and eigenvectors of the structure tensor, Jxx, Jxy, Jyy component of the structure tensor.

	"""
	# check if image_gradient_x and image_gradient_y are 2D arrays
	if len(image_gradient_x.shape) != 2 or len(image_gradient_y.shape) != 2:
		raise ValueError("image_gradient_x and image_gradient_y must be 2D arrays")
	# check if image_gradient_x and image_gradient_y have the same shape
	if image_gradient_x.shape != image_gradient_y.shape:
		raise ValueError("image_gradient_x and image_gradient_y must have the same shape")
	# check if local_sigma is a positive number
	if not isinstance(local_sigma, (float, int)) or local_sigma <= 0:
		raise ValueError("local_sigma must be a positive number")

	Jxx = scipy.ndimage.gaussian_filter(image_gradient_x * image_gradient_x, local_sigma, mode = 'nearest')
	Jxy = scipy.ndimage.gaussian_filter(image_gradient_x * image_gradient_y, local_sigma, mode = 'nearest')
	Jyy = scipy.ndimage.gaussian_filter(image_gradient_y * image_gradient_y, local_sigma, mode = 'nearest')

	Raw_Structure_Tensor = np.array([[Jxx, Jxy], [Jxy, Jyy]])

	Structure_Tensor = np.moveaxis(Raw_Structure_Tensor, [0, 1], [2, 3]) # For solving EigenProblem
	EigenValues, EigenVectors = np.linalg.eigh(Structure_Tensor) # eigh because matrix is symmetric 

	return Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy

########################################################################################

def make_vxvy(input_image, eigenvectors, threshold_value):
	"""
	Extracts the x and y components of the first eigenvector from the eigenvectors array.

	Parameters:
	input_image (numpy.ndarray): 2D array representing the input image.
	threshold_value (float): threshold value for the input image. Only pixels with intensity greater than or equal to threshold_value will be considered.

	Returns:
	tuple: Tuple containing 2D arrays representing x and y components of the first eigenvector. Pixels that do not meet the threshold condition are set to NaN.

	"""
	# check if input_image is 2D array
	if len(input_image.shape) != 2:
		raise ValueError("Input image must be a 2D array")
	# check if threshold_value is a number
	if not isinstance(threshold_value, (float, int)):
		raise ValueError("Threshold value must be a number")

	vx = eigenvectors[..., 0][:, :, 0]
	vx[input_image < threshold_value] = np.nan

	vy = eigenvectors[..., 0][:, :, 1]
	vy[input_image < threshold_value] = np.nan

	return vx, vy

########################################################################################

def convert_to_8bit_grayscale(filename):
	"""
	Read an image from a file using Pillow and return the 8-bit grayscale version of the image using OpenCV.
	If the image is already 8-bit grayscale, return the image without modification.

	Parameters:
	filename (str): The name of the file to be read, including the extension.

	Returns:
	numpy.ndarray: The 8-bit grayscale version of the image, represented as a numpy array.
	"""
	# Load the image using Pillow and convert to 8 bit
	img = Image.open(filename).convert("L")
	
	# Convert the image to a numpy array
	img = np.array(img)

	# Check if the image is 2D
	if img.ndim != 2:
		raise ValueError("Input image must be 2D grayscale.")

	# Normalize the image
	img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	return img

########################################################################################

def stitch_back_chunks(analyzed_chunk_list, padded_img, img, chunk_size):
	"""
	Reconstruct an image from a list of analyzed image chunks.

	Parameters:
		analyzed_chunk_list (list): A list of the analyzed image chunks.
		padded_img (numpy.ndarray): A padded version of the input image.
		img (numpy.ndarray): The original input image.
		chunk_size (int): The size of the chunks used for analyzing the image.

	Returns:
		numpy.ndarray: The reconstructed image.
	"""

	# Calculate the number of chunks in each dimension
	num_chunks = padded_img.shape[0] // chunk_size

	# Initialize a new NumPy array for the reconstructed image
	reconstructed_img = np.zeros((padded_img.shape))

	# Iterate over each chunk and copy it back to the correct location in the reconstructed image
	for i in range(len(analyzed_chunk_list)):
		row = i // num_chunks
		col = i % num_chunks

		chunk = analyzed_chunk_list[i]
		start_row = row * chunk_size
		end_row = start_row + chunk_size

		start_col = col * chunk_size
		end_col = start_col + chunk_size
		reconstructed_img[start_row:end_row, start_col:end_col] = chunk

	# Crop the reconstructed image to the size of the original input image
	reconstructed_img = reconstructed_img[:img.shape[0], :img.shape[1]]

	return reconstructed_img

########################################################################################