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

#######################################################################################################

import numpy as np
import scipy.ndimage
from numba import jit

@jit
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
	Jyy = scipy.ndimage.gaussian_filter(image_gradient_y * image_gradient_y, local_sigma, mode = 'nearest')
	Jxy = scipy.ndimage.gaussian_filter(image_gradient_x * image_gradient_y, local_sigma, mode = 'nearest')

	Raw_Structure_Tensor = np.array([[Jxx, Jxy], [Jxy, Jyy]])

	Structure_Tensor = np.moveaxis(Raw_Structure_Tensor, [0, 1], [2, 3]) # For solving EigenProblem
	EigenValues, EigenVectors = np.linalg.eigh(Structure_Tensor) # eigh because matrix is symmetric 

	return Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy
