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

import skimage.filters

def apply_gaussian_filter(raw_image, FilterKey):
	"""
	Applies a Gaussian filter to an image using scikit-image library.

	Parameters:
		raw_image (numpy.ndarray): The input image to be filtered.
		FilterKey (int or float): The standard deviation for Gaussian kernel.
		
	Returns:
		filtered_image (numpy.ndarray): The filtered image.
	"""
	if FilterKey <= 0:
		raise ValueError("FilterKey must be a positive number.")
	else:
		filtered_image = skimage.filters.gaussian(raw_image, sigma=FilterKey, mode='nearest', preserve_range=True)
		return filtered_image