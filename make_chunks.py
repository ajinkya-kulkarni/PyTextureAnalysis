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

def split_2d_array(array, chunk_size=1000):
    """
    Splits a 2D numpy array into smaller chunks of a specified size.

    Parameters
    ----------
    array : numpy.ndarray
        The 2D array to be split into chunks.
    chunk_size : int, optional
        The size of the chunks along each dimension (height and width), by default 1000.

    Returns
    -------
    list of numpy.ndarray
        A list of chunks, each of which is a 2D numpy array.

    Raises
    ------
    ValueError
        If `array` is not a 2D numpy array.
    ValueError
        If `chunk_size` is not a positive integer.
    """
    # Check if `array` is a 2D numpy array
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Input `array` must be a 2D numpy array.")

    # Check if `chunk_size` is a positive integer
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Input `chunk_size` must be a positive integer.")

    chunks = []
    height, width = array.shape
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            h = min(chunk_size, height - i)
            w = min(chunk_size, width - j)
            chunk = array[i:i + h, j:j + w]
            chunks.append(chunk)
    return chunks
