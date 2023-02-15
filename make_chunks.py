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

def split_into_chunks(img, chunk_size):
    """
    Splits a 2D grayscale image into chunks of a given size.

    Parameters:
        img (numpy.ndarray): The input 2D grayscale image.
        chunk_size (int): The size of the chunks to split
        the image into.
        overlap_pixels (int): The overlap between chunks,
        in pixels.

    Returns:
        list: A list of chunks, each of
        size chunk_size x chunk_size.
        numpy.ndarray: The padded image, with
        size padded_size x padded_size, where padded_size is a multiple of chunk_size.
    """
    # Divide the image into chunks
    chunks = []
    for i in range(0, img.shape[0] - chunk_size + 1, chunk_size):
        for j in range(0, img.shape[1] - chunk_size + 1, chunk_size ):
            chunk = img[i:i + chunk_size, j:j + chunk_size]
            chunks.append(chunk)

    return chunks
