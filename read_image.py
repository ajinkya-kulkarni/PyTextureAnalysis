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

from PIL import Image
import numpy as np
import cv2

def convert_to_8bit_grayscale(filename):
	"""
	Read an image from a file using Pillow and return the 8-bit grayscale version of the image using OpenCV.
	If the image is already 8-bit grayscale, return the image without modification.

	Parameters:
	filename (str): The name of the file to be read, including the extension.

	Returns:
	numpy.ndarray: The 8-bit grayscale version of the image, represented as a numpy array.
	"""
	# Load the image using Pillow and convert to 8 bit
	img = Image.open(filename)

	img = img.convert("L")
	
	# Convert the image to a numpy array
	img = np.array(img)

	# Check if the image is 2D
	if img.ndim != 2:
		raise ValueError("Input image must be 2D grayscale.")

	# Check if the image is already 8-bit
	if img.dtype == 'uint8':
		return img

	# Convert the image to 8-bit if necessary
	img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	return img
