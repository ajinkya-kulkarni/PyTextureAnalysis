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

from scipy import ndimage

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
