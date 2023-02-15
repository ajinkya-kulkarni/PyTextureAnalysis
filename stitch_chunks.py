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
from numba import jit

@jit
def stitch_back_chunks(analyzed_chunk_list, padded_img, img, chunk_size):
    """
    Reconstruct an image from a list of analyzed image chunks.

    Parameters:
        analyzed_chunk_list (list): A list of the analyzed image chunks.
        padded_img (numpy.ndarray): A padded version of the input image.
        img (numpy.ndarray): The original input image.
        chunk_size (int): The size of the chunks used for analyzing the image.

    Returns:
        numpy.ndarray: The reconstructed image.
    """

    # Calculate the number of chunks in each dimension
    num_chunks = padded_img.shape[0] // chunk_size

    # Initialize a new NumPy array for the reconstructed image
    reconstructed_img = np.zeros((padded_img.shape))

    # Iterate over each chunk and copy it back to the correct location in the reconstructed image
    for i in range(len(analyzed_chunk_list)):
        row = i // num_chunks
        col = i % num_chunks

        chunk = analyzed_chunk_list[i]
        start_row = row * chunk_size
        end_row = start_row + chunk_size

        start_col = col * chunk_size
        end_col = start_col + chunk_size
        reconstructed_img[start_row:end_row, start_col:end_col] = chunk

    # Crop the reconstructed image to the size of the original input image
    reconstructed_img = reconstructed_img[:img.shape[0], :img.shape[1]]

    return reconstructed_img
