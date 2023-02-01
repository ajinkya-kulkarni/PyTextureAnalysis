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