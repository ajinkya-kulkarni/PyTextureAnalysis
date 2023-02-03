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

from skimage.filters import threshold_mean

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