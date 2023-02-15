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
from numba import jit

@jit
def make_coherence(input_image, eigenvalues, Structure_Tensor, threshold_value):
	"""
	Calculate coherence values for a given input image, eigenvalues, structure tensor, and threshold value.

	Parameters:
	- input_image (numpy.ndarray): The input image for which coherence values are to be calculated.
	- eigenvalues (numpy.ndarray): The eigenvalues of the input image.
	- Structure_Tensor (numpy.ndarray): The structure tensor of the input image.
	- threshold_value (float): The threshold value to determine if the calculation should be done.

	Returns:
	- Coherence_Array (numpy.ndarray): An array containing the coherence values for the input image.

	"""

	Coherence_Array = np.full(input_image.shape, np.nan)

	for j in range(input_image.shape[1]):

		for i in range(input_image.shape[0]):

			# Check if the sum of the EigenValues of the Structure_Tensor is greater than 0

			if ( (input_image[i, j] >= threshold_value ) and ((eigenvalues[i, j].sum()) > 0) ) :

				trace = np.trace(Structure_Tensor[i, j])

				Smallest_Normalized_Eigenvalues = eigenvalues[i, j][0] / trace

				Largest_Normalized_Eigenvalues = eigenvalues[i, j][1] / trace

				Coherence_Array[i, j] = np.abs((Largest_Normalized_Eigenvalues -
												Smallest_Normalized_Eigenvalues) /
												(Smallest_Normalized_Eigenvalues +
												Largest_Normalized_Eigenvalues))

	return Coherence_Array
