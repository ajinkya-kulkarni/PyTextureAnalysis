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

def make_coherence(input_image, eigenvalues, threshold_value):
	"""
	Calculates the coherence of an image based on its eigenvalues and a threshold value.

	Parameters:
	- input_image (numpy array): 2D array representing the input image.
	- eigenvalues (numpy array): 3D array with shape (image.shape[0], image.shape[1], 2) representing the eigenvalues of the image.
	- threshold_value (float): Threshold value used to determine which elements of the input image will be used in the calculations.

	Returns:
	- coherence (numpy array): 2D array representing the coherence of the input image. Elements that do not meet the threshold condition are filled with NaN.
	"""
	if input_image.ndim != 2:
		raise ValueError("Input image must be 2-dimensional.")
	if eigenvalues.shape[:2] != input_image.shape:
		raise ValueError("Eigenvalues array must have the same shape as the input image.")
	if not isinstance(threshold_value, (float, int)):
		raise TypeError("Threshold value must be a float or an int.")

	mask = (input_image >= threshold_value) & ((eigenvalues[:,:,0] + eigenvalues[:,:,1]) > 0)

	coherence = np.abs((eigenvalues[:,:,1] - eigenvalues[:,:,0]) / (eigenvalues[:,:,0] + eigenvalues[:,:,1]))

	coherence[~mask] = np.nan

	return coherence