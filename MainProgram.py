#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
# Copyright (C) 2022 Bharti Arora <bharti.arora@mpinat.mpg.de>

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

########################################################################################

# This file contains the program to run this package in a standalone manner

########################################################################################

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

########################################################################################

from modules import *
from parameters import *

########################################################################################

print()

pbar = tqdm(total = 10, desc = 'Analyzing', colour = 'green')

# Read the image
raw_image = convert_to_8bit_grayscale(filename)

# Filter the image
filtered_image = make_filtered_image(raw_image, FilterKey)

pbar.update(1)

###########################

# Calculate local density by binarizing the image first (using simple mean thresholding), then convoluting it with a nxn kernel of ones. 
# Currently the kernel size is equal to the local window used for calculating coherence and orientation.
# Please refer to: https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-14-25718&id=477526 for more information.

# Binarize the image
binarized_image = binarize_image(filtered_image, radius = BinarizationKey)

pbar.update(1)

###########################

# Calculate the fibrotic_percentage area of the non-zero pixels compared to the image size
fibrotic_percentage = percentage_area(binarized_image)

pbar.update(1)

###########################

# Define the kernel and it's size
local_kernel_size = LocalDensityKey
if (local_kernel_size % 2 == 0):
	local_kernel_size = local_kernel_size + 1
if (local_kernel_size < 3):
	local_kernel_size = 3

local_kernel = np.ones((local_kernel_size, local_kernel_size), dtype = np.float32) / (local_kernel_size * local_kernel_size)

Local_Density = convolve(raw_image, local_kernel)

Local_Density = np.divide(Local_Density, Local_Density.max(), out=np.full(Local_Density.shape, np.nan), where=Local_Density.max() != 0)

pbar.update(1)

###########################

# Calculate image gradients in X and Y directions
image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

pbar.update(1)

###########################

# Calculate the structure tensor and solve for EigenValues, EigenVectors
Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

pbar.update(1)

###########################

# Calculate Coherence

Image_Coherance = make_coherence(filtered_image, EigenValues, Structure_Tensor, ThresholdValueKey)

pbar.update(1)

###########################

# Calculate Orientation
Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)
vx, vy = make_vxvy(filtered_image, EigenVectors, ThresholdValueKey)

pbar.update(1)

###########################

# Plot the results

fig = make_mosiac_plot(raw_image, binarized_image, filtered_image, Local_Density, Image_Coherance, Image_Orientation, vx, vy, filename, LocalSigmaKey, fibrotic_percentage, SpacingKey, ScaleKey, FIGSIZE, DPI, PAD, FONTSIZE_TITLE, pad_fraction, aspect)

saving_name = 'Results' + filename + '_LocalSigma_' + str(LocalSigmaKey) + '.png'
plt.savefig(saving_name)
plt.close()

pbar.update(1)

###########################

# Perform statistical analysis

perform_statistical_analysis(filename, LocalSigmaKey, Image_Orientation, Image_Coherance)

pbar.update(1)

###########################

pbar.close()

print()

###########################