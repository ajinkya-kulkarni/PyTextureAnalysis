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

###################################################################################

import numpy as np

def generate_padded_image(img, chunk_size):
    """
    Generate a padded image that is square and a multiple of the chunk size.

    Parameters:
        img (numpy.ndarray): The input image.
        chunk_size (int): The size of the chunks used for analyzing the image.

    Returns:
        numpy.ndarray: The padded image.
    """

    # Pad the image to make it square and a multiple of chunk_size
    max_size = max(img.shape)
    padded_size = max_size + (chunk_size - max_size % chunk_size) % chunk_size

    padded_img = np.zeros((padded_size, padded_size))
    padded_img[:img.shape[0], :img.shape[1]] = img

    return padded_img
