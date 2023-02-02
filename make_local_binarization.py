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

from skimage import filters
import numpy as np

def binarize_image_with_local_otsu(image, n):
	"""
	Calculates a binarized image using the method skimage.filters.threshold_local,
	where the local offset is determined by a local Otsu method. 

	Parameters:
	image: numpy array, input image
	n: integer, block size for local threshold calculation

	Returns:
	binary_image: numpy array, binarized image
	"""
	# Check if the block size is odd
	if n % 2 == 0:
		raise ValueError("Block size should be odd.")
		
	# Check if the input image is 2D
	if len(image.shape) != 2:
		raise ValueError("Input should be a 2D image.")

	# Create a block view of the image
	block_view = view_as_blocks(image, (n, n))

	# Calculate local Otsu threshold value within each block
	otsu_thresholds = np.array([filters.threshold_otsu(block) for block in block_view.reshape(-1, n, n)])

	# Use the local Otsu threshold values as the offset for threshold_local
	local_threshold = filters.threshold_local(image, block_size=n, method='gaussian', offset=otsu_thresholds.mean())

	# Binarize the image using threshold_local with the calculated local threshold
	binary_image = image > local_threshold

	return binary_image


## Example usage:

# import numpy as np
# import skimage.filters as filters
# from skimage.filters import threshold_local

# # Create a sample image
# image = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [6, 7, 8, 9, 10], [10, 9, 8, 7, 6], [1, 2, 3, 4, 5]])

# # Binarize the image with a block size of 3
# n = 3
# binarized_image = binarize_image_with_local_otsu(image, n)

# print(binarized_image)
