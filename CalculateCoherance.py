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

def make_coherence(input_image, eigenvalues, structure_tensor, threshold_value):
	"""
	Calculates coherence of an image using eigenvalues and structure tensor.

	Parameters:
	input_image (numpy.ndarray): 2D array representing the input image.
	eigenvalues (numpy.ndarray): 2D array of eigenvalues of the structure tensor.
	structure_tensor (numpy.ndarray): 2D array of the structure tensor.
	threshold_value (float): threshold value for the input image. Only pixels with intensity greater than or equal to threshold_value will be considered.

	Returns:
	numpy.ndarray: 2D array representing the coherence of the input image. Pixels that do not meet the threshold condition are set to NaN.

	"""
	# check if input_image is 2D array
	if len(input_image.shape) != 2:
		raise ValueError("Input image must be a 2D array")
	# check if eigenvalues and structure_tensor has the same shape as input_image
	if eigenvalues.shape != input_image.shape or structure_tensor.shape != input_image.shape:
		raise ValueError("Eigenvalues and Structure Tensor must have the same shape as input image")
	# check if threshold_value is a number
	if not isinstance(threshold_value, (float, int)):
		raise ValueError("Threshold value must be a number")

	coherence = np.zeros(input_image.shape)
	mask = (input_image >= threshold_value) & ((eigenvalues.sum(axis=-1)) > 0)
	trace = np.trace(structure_tensor, axis=-1)

	smallest_normalized_eigenvalues = eigenvalues[..., 0] / trace
	largest_normalized_eigenvalues = eigenvalues[..., 1] / trace

	coherence[mask] = np.abs((largest_normalized_eigenvalues - smallest_normalized_eigenvalues) / (smallest_normalized_eigenvalues + largest_normalized_eigenvalues))[mask]

	coherence[~mask] = np.nan

	return coherence