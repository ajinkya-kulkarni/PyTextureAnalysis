#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
# Copyright (C) 2022 Bharti Arora <bharti.arora@mpinat.mpg.de>

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
import pandas as pd

from scipy import ndimage
from scipy.stats import circmean, circstd
from scipy import signal

from skimage.filters import threshold_mean, gaussian
from skimage.morphology import disk
from skimage.filters import rank

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size

########################################################################################

def binarize_image(image, radius = 15):
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

	image = image.astype('uint8')

	selem = disk(radius)
	threshold_value = rank.otsu(image, selem)

	# threshold_value = threshold_mean(image)

	binary_image = image > threshold_value

	return binary_image

########################################################################################

def make_filtered_image(input_image, filter_sigma):
	"""
	Applies a Gaussian filter to an input image.

	Parameters:
	input_image (ndarray): a NumPy array representing the input image to be filtered.
	filter_sigma (float): a numeric value representing the standard deviation of the Gaussian filter.
	filter_mode (str): a string representing the mode of the Gaussian filter (default: 'nearest').
	preserve_range (bool): a boolean value indicating whether to preserve the data range of the input image (default: True).

	Returns:
	filtered_image (ndarray): a NumPy array representing the filtered image.
	"""
	filtered_image = gaussian(input_image, sigma = filter_sigma, mode = 'nearest', preserve_range = True)

	return filtered_image

########################################################################################

def percentage_area(image):
	"""
	Calculate the percentage of nonzero pixels in a 2D binary image.

	Parameters:
		image (np.array): Input image to be quantified.

	Returns:
		np.array: Returns a value corresponding to the percentage of nonzero pixels

	Raises:
		ValueError: If the input image is not 2D or not binary.
	"""
	# Check if the input image is 2D
	if len(image.shape) != 2:
		raise ValueError("Input should be a 2D image.")

	# Check if the input image is binary
	if len(np.unique(image)) != 2:
		raise ValueError("Input should be a 2D binarized image.")

	non_zero_pixels = np.count_nonzero(image)

	image_size = image.shape[0] * image.shape[1]

	percentage_area = np.round(100 * non_zero_pixels / image_size, 1)

	return percentage_area

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

	# Convert the input image to a valid data type
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
	image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

	# Get the total number of elements in the kernel
	total_elements = k_h * k_w

	# Perform convolution
	result = signal.convolve2d(image, kernel / total_elements, mode='same', boundary='fill', fillvalue=0)

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

	Orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) * 180 / np.pi
	Orientation[Orientation < 0] += 180
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

	Jxx = ndimage.gaussian_filter(image_gradient_x * image_gradient_x, local_sigma, mode = 'nearest')
	Jxy = ndimage.gaussian_filter(image_gradient_x * image_gradient_y, local_sigma, mode = 'nearest')
	Jyy = ndimage.gaussian_filter(image_gradient_y * image_gradient_y, local_sigma, mode = 'nearest')

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
	Read an image from a file using Pillow and return the 8-bit grayscale version of the image using NumPy.
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
	img = (img - np.min(img)) * (255 / (np.max(img) - np.min(img)))
	img = img.astype(np.uint8)

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

def perform_statistical_analysis(filename, LocalSigmaKey, Image_Orientation, Image_Coherance):
	"""
	Calculate various statistical parameters of the given image data and save the results in a CSV file.

	Parameters:
	filename (str): The name of the file being processed.
	LocalSigmaKey (int): The value of LocalSigmaKey used in the image processing.
	Image_Orientation (numpy array): An array of orientation data for the image.
	Image_Coherance (numpy array): An array of coherence data for the image.

	Returns:
	None.
	"""
	
	# Convert Image_Orientation to radians
	Image_Orientation_rad = np.deg2rad(Image_Orientation)

	# Calculate circular and normal means of Image_Orientation
	CircMean = circmean(Image_Orientation_rad[~np.isnan(Image_Orientation_rad)], low=0, high=np.pi)
	NormalMean = np.nanmean(Image_Orientation_rad)

	# Calculate circular and normal standard deviations of Image_Orientation
	CircStdDev = circstd(Image_Orientation_rad[~np.isnan(Image_Orientation_rad)], low=0, high=np.pi)
	NormalStdDev = np.nanstd(Image_Orientation_rad)

	# Calculate circular variance of Image_Orientation
	CircVar = circular_variance(Image_Orientation_rad)

	# Calculate low and high coherance values
	Image_Coherance_temp = Image_Coherance[~np.isnan(Image_Coherance)].copy()
	histogram_coherance = plt.hist(Image_Coherance_temp, bins=2, weights=np.ones(len(Image_Coherance_temp))/len(Image_Coherance_temp))
	plt.close()
	low_coherance, high_coherance = np.round(100 * histogram_coherance[0], 2)

	# Combine the results into a single numpy array
	results_array = np.asarray((filename, np.round(NormalMean, 2), np.round(CircMean, 2), np.round(NormalStdDev, 2), np.round(CircStdDev, 2), np.round(CircVar, 2), np.round(np.nanmean(Image_Coherance_temp), 2), np.round(np.nanmedian(Image_Coherance_temp), 2), np.round(np.nanstd(Image_Coherance_temp), 2), low_coherance, high_coherance))
	
	results_array = np.atleast_2d(results_array)

	return results_array

########################################################################################

def make_mosiac_plot(raw_image, binarized_image, filtered_image, Local_Density, Image_Coherance, Image_Orientation, vx, vy, filename, LocalSigmaKey, percentage, SpacingKey, ScaleKey, FIGSIZE, DPI, PAD, FONTSIZE_TITLE, pad_fraction, aspect):
	"""
	Creates a mosaic plot of an image analysis with six subplots representing different aspects of the image.

	Parameters:
	raw_image (ndarray): the original image as a NumPy array.
	binarized_image (ndarray): a binary version of the image as a NumPy array.
	filtered_image (ndarray): a filtered version of the image as a NumPy array.
	Local_Density (ndarray): a density map of the image as a NumPy array.
	Image_Coherance (ndarray): a coherence map of the image as a NumPy array.
	Image_Orientation (ndarray): an orientation map of the image as a NumPy array.
	vx (ndarray): x-component of the local orientation vector as a NumPy array.
	vy (ndarray): y-component of the local orientation vector as a NumPy array.
	filename (str): a string representing the name of the file being analyzed.
	LocalSigmaKey (float): a numeric value representing the local sigma of the filter used.
	percentage (float): a numeric value representing the percentage of fibrotic tissue in the image.
	SpacingKey (int): a numeric value representing the spacing of the local orientation vectors.
	ScaleKey (float): a numeric value representing the scale of the local orientation vectors.
	FIGSIZE (tuple): a tuple representing the size of the figure in inches.
	DPI (int): a numeric value representing the resolution of the figure in dots per inch.
	PAD (float): a numeric value representing the padding of the subplot titles.
	FONTSIZE_TITLE (int): a numeric value representing the font size of the subplot titles.
	pad_fraction (float): a numeric value representing the fraction of the padding for the colorbar.
	aspect (float): a numeric value representing the aspect ratio of the subplots.

	Returns:
	Fig object.

	Returns a figure with six subplots arranged in a 2x3 grid that represent different aspects of an image analysis. Each subplot has a colorbar that shows the color scale for that particular subplot.
	"""

	fig, axes = plt.subplot_mosaic("ABC;DEF", figsize=FIGSIZE, constrained_layout=True, dpi=DPI)

	###########################

	im = axes["A"].imshow(raw_image, vmin = 0, vmax = 255, cmap = 'binary')
	axes["A"].set_title('Uploaded Image', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["A"].set_xticks([])
	axes["A"].set_yticks([])

	divider = make_axes_locatable(axes["A"])
	width = axes_size.AxesY(axes["A"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = plt.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)
	cbar.remove()

	###########################

	im = axes["B"].imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'binary')
	axes["B"].set_title('Filtered Image', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["B"].set_xticks([])
	axes["B"].set_yticks([])

	divider = make_axes_locatable(axes["B"])
	width = axes_size.AxesY( axes["B"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = plt.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)
	cbar.remove()

	###########################

	im =  axes["C"].imshow(Local_Density, vmin = 0, vmax = 1, cmap = 'Spectral_r')

	axes["C"].set_title('Local Density, ' + str(percentage) + '% fibrotic', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["C"].set_xticks([])
	axes["C"].set_yticks([])

	divider = make_axes_locatable(axes["C"])
	width = axes_size.AxesY(axes["C"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = plt.colorbar(im, cax=cax)
	cbar.formatter.set_powerlimits((0, 0))
	# cbar.formatter.set_useMathText(True)
	cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

	###########################

	im = axes["D"].imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'Spectral_r')
	# im = axes["D"].imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.Spectral_r(Image_Coherance), vmin = 0, vmax = 1, cmap = 'Spectral_r')
	# im = axes["D"].imshow((raw_image/raw_image.max()) * (Image_Coherance), vmin = 0, vmax = 1, cmap = 'Spectral_r')

	axes["D"].set_title('Coherence', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["D"].set_xticks([])
	axes["D"].set_yticks([])

	divider = make_axes_locatable(axes["D"])
	width = axes_size.AxesY(axes["D"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = plt.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

	###########################

	im = axes["E"].imshow(Image_Orientation/180, vmin = 0, vmax = 1, cmap = 'hsv')
	# im = axes["E"].imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.hsv(Image_Orientation/180), vmin = 0, vmax = 1, cmap = 'hsv')
	# im = axes["E"].imshow((raw_image/raw_image.max()) * (Image_Orientation/180), vmin = 0, vmax = 1, cmap = 'hsv')

	axes["E"].set_title('Orientation', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["E"].set_xticks([])
	axes["E"].set_yticks([])

	divider = make_axes_locatable(axes["E"])
	width = axes_size.AxesY(axes["E"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = fig.colorbar(im, cax = cax, ticks = np.linspace(0, 1, 5))
	cbar.ax.set_yticklabels([r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', r'$135^{\circ}$', r'$180^{\circ}$'])
	ticklabs = cbar.ax.get_yticklabels()
	cbar.ax.set_yticklabels(ticklabs, fontsize = FONTSIZE_TITLE)

	###########################

	im = axes["F"].imshow(raw_image, vmin = 0, vmax = 255, cmap = 'Oranges', alpha = 0.8)

	xmesh, ymesh = np.meshgrid(np.arange(raw_image.shape[0]), np.arange(raw_image.shape[1]), indexing = 'ij')

	axes["F"].quiver(ymesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], xmesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vy[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vx[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
	scale = ScaleKey, headlength = 0, headaxislength = 0, 
	pivot = 'middle', color = 'black', angles = 'xy')

	axes["F"].set_title('Local Orientation', pad = PAD, fontsize = FONTSIZE_TITLE)
	axes["F"].set_xticks([])
	axes["F"].set_yticks([])

	divider = make_axes_locatable(axes["F"])
	width = axes_size.AxesY(axes["F"], aspect=1./aspect)
	pad = axes_size.Fraction(pad_fraction, width)
	cax = divider.append_axes("right", size=width, pad=pad)
	cbar = plt.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)
	cbar.remove()

	###########################

	return fig

########################################################################################

def load_pandas_dataframe(results_array):
	"""
	Creates a Pandas DataFrame from a 2D NumPy array.

	Parameters
	----------
	results_array : numpy.ndarray
		A 2D array of shape (n, 10) containing the results of a computation.

	Returns
	-------
	pandas.DataFrame
		A DataFrame with 11 columns, containing the following data from results_array:
			- Name of the uploaded image
			- Mean Orientation
			- Circular Mean Orientation
			- StdDev Orientation
			- Circular StdDev Orientation
			- Circular Variance
			- Mean Coherance
			- Median Coherance
			- StdDev Coherance
			- % Low Coherance
			- % High Coherance
	"""
	dataframe =  pd.DataFrame({
		"Uploaded Image": results_array[:, 0],
		"Mean Orientation": results_array[:, 1],
		"Circular Mean Orientation": results_array[:, 2],
		"StdDev Orientation": results_array[:, 3],
		"Circular StdDev Orientation": results_array[:, 4],
		"Circular Variance": results_array[:, 5],
		"Mean Coherance": results_array[:, 6],
		"Median Coherance": results_array[:, 7],
		"StdDev Coherance": results_array[:, 8],
		"% Low Coherance": results_array[:, 9],
		"% High Coherance": results_array[:, 10]})

	return dataframe

########################################################################################