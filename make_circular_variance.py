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

def circular_variance(angles):
	"""
	Calculates circular variance of given angles

	Parameters:
	angles (numpy.ndarray): 1D array of angles in radians

	Returns:
	float: circular variance value
	"""
	# check if angles is a 1D array
	if len(angles.shape) != 1:
		raise ValueError("Input must be a 1D array of angles in radians")
	#remove NaN values
	angles = angles[~np.isnan(angles)]
	length = angles.size
	# check if the input has at least one valid value 
	if length == 0:
		raise ValueError("Input must contain at least one valid value")
	# calculate circular variance
	S = np.sum(np.sin(angles))
	C = np.sum(np.cos(angles))
	R = np.sqrt(S**2 + C**2)
	R_avg = R/length
	V = 1 - R_avg
	return V
