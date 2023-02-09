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

import cv2
import numpy as np
from numba import jit

@jit
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