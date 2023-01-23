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

#########################################################################################################

import numpy as np
import cv2 as cv

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

	Orientation = 0.5 * ( cv.phase( (Jyy - Jxx), (2 * Jxy), angleInDegrees = True) )

	Orientation[input_image < threshold_value] = np.nan

	return Orientation