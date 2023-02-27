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
import cv2 as cv
import skimage as skimage
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import os
import time

import sys

# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 

# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

########################################################################################

from modules import *
from parameters import *

########################################################################################

# Read the image
raw_image = convert_to_8bit_grayscale(filename)

# Filter the image
filtered_image = skimage.filters.gaussian(raw_image, sigma = FilterKey, mode = 'nearest', preserve_range = True)

###########################

# Calculate local density by binarizing the image first (using simple mean thresholding), then convoluting it with a nxn kernel of ones. 
# Currently the kernel size is equal to the local window used for calculating coherence and orientation.
# Please refer to: https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-14-25718&id=477526 for more information.

# Binarize the image
binarized_image = binarize_image(filtered_image, radius = BinarizationKey)

###########################

# Calculate the percentage area of the non-zero pixels
percentage = percentage_area(binarized_image)

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

###########################

# Calculate image gradients in X and Y directions
image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

###########################

# Calculate the structure tensor and solve for EigenValues, EigenVectors
Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

###########################

# Calculate Coherence

Image_Coherance = make_coherence(filtered_image, EigenValues, Structure_Tensor, ThresholdValueKey)

###########################

# Calculate Orientation
Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)
vx, vy = make_vxvy(filtered_image, EigenVectors, ThresholdValueKey)

###########################

# Plot the results

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
im = plt.imshow(raw_image, vmin = 0, vmax = 255, cmap = 'binary')

plt.title('Uploaded Image', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

plt.savefig('Uploaded_Image.png')
plt.close()

###########################

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
# 			im = plt.imshow(binarized_image, vmin = 0, vmax = 1, cmap = 'binary')
im = plt.imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'binary')
plt.title('Filtered Image', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

plt.savefig('Filtered_Image.png')
plt.close()

###########################

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
im = plt.imshow(Local_Density, vmin = 0, vmax = 1, cmap = 'Spectral_r')

plt.title('Local Density, ' + str(percentage) + '% fibrotic', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = plt.colorbar(im, cax=cax)
cbar.formatter.set_powerlimits((0, 0))
# cbar.formatter.set_useMathText(True)
cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

plt.savefig('Local_Density.png')
plt.close()

###########################

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

im = plt.imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'Spectral_r')
# im = plt.imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.Spectral_r(Image_Coherance), vmin = 0, vmax = 1, cmap = 'Spectral_r')
# im = plt.imshow((raw_image/raw_image.max()) * (Image_Coherance), vmin = 0, vmax = 1, cmap = 'Spectral_r')

plt.title('Coherence', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

plt.savefig('Coherence.png')
plt.close()

###########################

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

im = plt.imshow(Image_Orientation/180, vmin = 0, vmax = 1, cmap = 'hsv')
# im = plt.imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.hsv(Image_Orientation/180), vmin = 0, vmax = 1, cmap = 'hsv')
# im = plt.imshow((raw_image/raw_image.max()) * (Image_Orientation/180), vmin = 0, vmax = 1, cmap = 'hsv')

plt.title('Orientation', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = fig.colorbar(im, cax = cax, ticks = np.linspace(0, 1, 5))
cbar.ax.set_yticklabels([r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', r'$135^{\circ}$', r'$180^{\circ}$'])
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize = FONTSIZE_TITLE)

plt.savefig('Orientation.png')
plt.close()

###########################

fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

im = plt.imshow(raw_image, vmin = 0, vmax = 255, cmap = 'Oranges', alpha = 0.8)

xmesh, ymesh = np.meshgrid(np.arange(raw_image.shape[0]), np.arange(raw_image.shape[1]), indexing = 'ij')

plt.quiver(ymesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], xmesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vy[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vx[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
scale = ScaleKey, headlength = 0, headaxislength = 0, 
pivot = 'middle', color = 'black', angles = 'xy')

plt.title('Local Orientation', pad = PAD, fontsize = FONTSIZE_TITLE)
plt.xticks([])
plt.yticks([])

ax = plt.gca()
divider = make_axes_locatable(ax)
width = axes_size.AxesY(ax, aspect=1./aspect)
pad = axes_size.Fraction(pad_fraction, width)
cax = divider.append_axes("right", size=width, pad=pad)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

plt.savefig('Local_Orientation.png')
plt.close()

###########################