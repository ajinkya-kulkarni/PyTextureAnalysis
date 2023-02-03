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
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_properties(img, angles):
	"""
	Calculate the contrast, correlation, energy, and homogeneity of a grayscale image using the gray-level co-occurrence matrix (GLCM).

	Parameters:
	-----------
	img : numpy.ndarray
		An 8-bit grayscale image.

	Returns:
	--------
	contrast : float
		A measure of the difference between the intensities of pairs of pixels in an image.
		Higher values of contrast indicate a greater difference between the intensities of pairs of pixels, while lower values of contrast indicate that the intensities of pairs of pixels are more similar.
		
	correlation : float
		A measure of the linear relationship between the intensities of pairs of pixels in an image.
		Higher values of correlation indicate that there is a strong linear relationship between the intensities of pairs of pixels, while lower values of correlation indicate that the intensities of pairs of pixels are not strongly related.
		
	energy : float
		A measure of the uniformity of the gray levels in an image.
		Higher values of energy indicate that the gray levels are more uniform, while lower values of energy indicate that the gray levels are more heterogeneous.
		
	homogeneity : float
		A measure of the similarity of the intensities of pairs of pixels in an image.
		Higher values of homogeneity indicate that the intensities of pairs of pixels are more similar, while lower values of homogeneity indicate that the intensities of pairs of pixels are more dissimilar.
	"""
	# Check if image is grayscale
	if len(img.shape) != 2:
		raise ValueError("Image must be grayscale")

	# Check if image has 8-bit resolution
	if img.dtype != np.uint8:
		img = img.astype(np.uint8)	

	# Calculate the GLCM for the defined angles
	glcm = graycomatrix(img, [1], angles, 255, symmetric = True, normed = True)

	# Check if the GLCM has been properly calculated
	if glcm.size == 0:
		print("Error: Could not calculate the GLCM")
		exit()

	# Calculate the properties of the GLCM
	contrast = graycoprops(glcm, 'contrast')
	correlation = graycoprops(glcm, 'correlation')
	energy = graycoprops(glcm, 'energy')
	homogeneity = graycoprops(glcm, 'homogeneity')

	return contrast, correlation, energy, homogeneity