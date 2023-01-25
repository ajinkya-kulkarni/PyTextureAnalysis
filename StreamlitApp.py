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

import streamlit as st

import numpy as np
import cv2 as cv
import skimage as skimage
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import time
from io import BytesIO

import sys

# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 

# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

#######################################################################################################

from make_coherence import *
from make_image_gradients import *
from make_orientation import *
from make_structure_tensor_2d import *
from make_vxvy import *

#######################################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PyTextureAnalysis', page_icon = image_bytes, layout = "wide", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PyTextureAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

FONTSIZE = 23
DPI = 300
FACTOR = 1.2

# Title of the web app

st.title(':blue[Application for demonstrating the PyTextureAnalysis package]')

st.markdown("")

########################################################################################

def main():

	uploaded_file = st.file_uploader("Upload an 8 bit grayscale image to be analyzed:", type=["tif", "tiff"], accept_multiple_files = False, label_visibility = 'visible')

	st.markdown("""---""")

	if uploaded_file is None:
		st.stop()

	####################################################################################

	raw_image_from_pillow = Image.open(uploaded_file)

	raw_image = np.array(raw_image_from_pillow)

	if len(raw_image.shape) != 2:
		raise ValueError("Invalid image format, expected 2D image but got {}D image".format(raw_image.shape))

	####################################################################################

	st.slider('Filter', min_value = 0.1, max_value = 5.0, value = 1.0, step = 0.1, format = '%0.1f', label_visibility = "visible", key = '-FilterKey-')
	FilterKey = float(st.session_state['-FilterKey-'])

	filtered_image = skimage.filters.gaussian(raw_image, sigma = FilterKey, mode = 'nearest', preserve_range = True)

	####################################################################################

	fig, ax = plt.subplots(1, 2, figsize = (25, 10), sharex = True, sharey = True)

	ax[0].imshow(raw_image, vmin = 0, vmax = 255, cmap = 'viridis')
	ax[0].set_title('Uploaded Image', pad = 30, fontsize = FONTSIZE)
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	ax[1].imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'viridis')
	ax[1].set_title('Filtered Image', pad = 30, fontsize = FONTSIZE)
	ax[1].set_xticks([])
	ax[1].set_yticks([])

	fig.tight_layout()
	st.pyplot(fig)

	####################################################################################

	st.markdown("")

	####################################################################################

	st.slider('Local Sigma', min_value = 1, max_value = 20, value = 10, step = 1, format = '%d', label_visibility = "visible", key = '-LocalSigmaKey-')
	LocalSigmaKey = int(st.session_state['-LocalSigmaKey-'])

	st.slider('Threshold Value', min_value = 5, max_value = 200, value = 40, step = 1, format = '%d', label_visibility = "visible", key = '-ThresholdValueKey-')
	ThresholdValueKey = int(st.session_state['-ThresholdValueKey-'])

	st.slider('Spacing', min_value = 5, max_value = 50, value = 20, step = 1, format = '%d', label_visibility = "visible", key = '-SpacingKey-')
	SpacingKey = int(st.session_state['-SpacingKey-'])

	st.slider('Scale', min_value = 10, max_value = 100, value = 60, step = 1, format = '%d', label_visibility = "visible", key = '-ScaleKey-')
	ScaleKey = int(st.session_state['-ScaleKey-'])

	st.slider('Alpha', min_value = 0.1, max_value = 1.0, value = 0.7, step = 0.1, format = '%0.1f', label_visibility = "visible", key = '-AlphaKey-')
	AlphaKey = float(st.session_state['-AlphaKey-'])

	####################################################################################

	try:

		image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

		Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

		Image_Coherance = make_coherence(filtered_image, EigenValues, ThresholdValueKey)

		Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)

		vx, vy = make_vx_vy(filtered_image, EigenVectors, ThresholdValueKey)

	except:

		raise Exception('Something went wrong in the analysis')

	####################################################################################

	fig, ax = plt.subplots(1, 3, figsize = (40, 12), sharex = True, sharey = True)

	im1 = ax[0].imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'RdYlBu_r')

	divider = make_axes_locatable(ax[0])
	cax = divider.append_axes("right", size="5%", pad = 0.4)
	cbar = fig.colorbar(im1, cax = cax, ticks = np.linspace(0, 1, 5))
	cbar.ax.set_yticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize = FACTOR*FONTSIZE)

	ax[0].set_title('Coherance', pad = 30, fontsize = FACTOR*FONTSIZE)
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	##################

	im2 = ax[1].imshow(Image_Orientation/180, vmin = 0, vmax = 1, cmap = 'hsv')

	divider = make_axes_locatable(ax[1])
	cax = divider.append_axes("right", size="5%", pad=0.4)
	cbar = fig.colorbar(im2, cax = cax, ticks = np.linspace(0, 1, 5))
	cbar.ax.set_yticklabels([r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', r'$135^{\circ}$', r'$180^{\circ}$'], fontsize = FACTOR*FONTSIZE)

	ax[1].set_title('Orientation', pad = 30, fontsize = FACTOR*FONTSIZE)
	ax[1].set_xticks([])
	ax[1].set_yticks([])

	##################

	im3 = ax[2].imshow(raw_image, cmap = 'Oranges', alpha = AlphaKey)

	xmesh, ymesh = np.meshgrid(np.arange(raw_image.shape[0]), np.arange(raw_image.shape[1]), indexing = 'ij')

	ax[2].quiver(ymesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], 
				xmesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
				vy[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], 
				vx[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
				scale = ScaleKey, headlength = 0, headaxislength = 0, 
				pivot = 'middle', color = 'k', angles = 'xy')

	ax[2].set_title('Local Orientation', pad = 30, fontsize = FACTOR*FONTSIZE)
	ax[2].set_xticks([])
	ax[2].set_yticks([])

	divider = make_axes_locatable(ax[2])
	cax = divider.append_axes("right", size="5%", pad=0.4)
	cax.remove()

	fig.tight_layout()
	st.pyplot(fig)

	########################################################################

	st.stop()

if __name__== "__main__":
	main()