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

input_image, eigenvalues, threshold_value

def make_coherence(input_image, eigenvalues, Structure_Tensor, threshold_value):

	Coherance_Array = np.zeros(input_image.shape)

	vx = np.zeros(input_image.shape)

	vy = np.zeros(input_image.shape)

	### Calculate Coherance and Orientation vector field

	Coherance_Array = np.zeros(input_image.shape)

	#############################################

	for j in range(input_image.shape[1]):

		for i in range(input_image.shape[0]):
			#############################################

			### Calculate Coherance

			if ( (eigenvalues[i, j].sum()) > 0):

				Smallest_Normalized_Eigenvalues = eigenvalues[i, j][0] / np.trace(Structure_Tensor[i, j])

				Largest_Normalized_Eigenvalues = eigenvalues[i, j][1] / np.trace(Structure_Tensor[i, j])

				Coherance_Array[i, j] = np.abs((Largest_Normalized_Eigenvalues -
												Smallest_Normalized_Eigenvalues) /
												(Smallest_Normalized_Eigenvalues +
												Largest_Normalized_Eigenvalues))

			else:

				Coherance_Array[i, j] = np.nan
