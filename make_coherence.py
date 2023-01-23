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
	threshold_value (float): threshold value for the input image. Only pixels with intensity greater than or equal to threshold_value will be considered.

	Returns:
	numpy.ndarray: 2D array representing the coherence of the input image. Pixels that do not meet the threshold condition are set to NaN.

	"""

	coherence = np.zeros(input_image.shape)

	#############################################

	for j in range(input_image.shape[1]):

		for i in range(input_image.shape[0]):

			if ( (input_image[i, j] >= threshold_value ) and ((eigenvalues[i, j].sum()) > 0) ) :

				Smallest_Normalized_Eigenvalues = eigenvalues[i, j][0] / np.trace(structure_tensor[i, j])

				Largest_Normalized_Eigenvalues = eigenvalues[i, j][1] / np.trace(structure_tensor[i, j])

				coherence[i, j] = np.abs((Largest_Normalized_Eigenvalues - Smallest_Normalized_Eigenvalues) / (Smallest_Normalized_Eigenvalues + Largest_Normalized_Eigenvalues))

			else:

				coherence[i, j] = np.nan

	return coherence