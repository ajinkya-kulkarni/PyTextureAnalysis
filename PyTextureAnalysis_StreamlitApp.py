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

########################################################################################

import streamlit as st

import numpy as np
import cv2 as cv
import skimage as skimage
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import os
import time
from io import BytesIO

import sys

# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 

# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

########################################################################################

from make_coherence import *
from make_image_gradients import *
from make_orientation import *
from make_structure_tensor_2d import *
from make_vxvy import *

########################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PyTextureAnalysis', page_icon = image_bytes, layout = "wide", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PyTextureAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

# Title of the web app

st.title(':blue[Texture analysis using PyTextureAnalysis]')
st.caption('For more information or to give feedback, visit https://github.com/ajinkya-kulkarni/PyTextureAnalysis', unsafe_allow_html = False)

st.markdown("")

########################################################################################

PAD = 10
FONTSIZE_TITLE = 15
DPI = 500

########################################################################################

with st.form(key = 'form1', clear_on_submit = False):

	uploaded_file = st.file_uploader("Upload a 2D grayscale image to be analyzed. Works best with images with the same X and Y dimensions.", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'visible')

	st.markdown("""---""")
	
	left_column1, middle_column1, right_column1  = st.columns(3)

	with left_column1:
		st.slider('Gaussian image filter sigma [pixels]', min_value = 0.1, max_value = 5.0, value = 1.0, step = 0.1, format = '%0.1f', label_visibility = "visible", key = '-FilterKey-')
		FilterKey = float(st.session_state['-FilterKey-'])

	with middle_column1:
		st.slider('Gaussian local window [pixels]', min_value = 2, max_value = 50, value = 10, step = 2, format = '%d', label_visibility = "visible", key = '-LocalSigmaKey-')
		LocalSigmaKey = int(st.session_state['-LocalSigmaKey-'])

	with right_column1:
		st.slider('Threshold value for pixel evaluation [pixels]' , min_value = 5, max_value = 200, value = 20, step = 1, format = '%d', label_visibility = "visible", key = '-ThresholdValueKey-')
		ThresholdValueKey = int(st.session_state['-ThresholdValueKey-'])

	####################################################################################

	left_column2, middle_column2, right_column2  = st.columns(3)

	with left_column2:
		st.slider('Spacing between the orientation vectors', min_value = 5, max_value = 50, value = 20, step = 1, format = '%d', label_visibility = "visible", key = '-SpacingKey-')
		SpacingKey = int(st.session_state['-SpacingKey-'])

	with middle_column2:
		st.slider('Length of the orientation vectors', min_value = 10, max_value = 100, value = 60, step = 1, format = '%d', label_visibility = "visible", key = '-ScaleKey-')
		ScaleKey = int(st.session_state['-ScaleKey-'])

	with right_column2:
		st.slider('Alpha value for image transparency', min_value = 0.1, max_value = 1.0, value = 0.7, step = 0.1, format = '%0.1f', label_visibility = "visible", key = '-AlphaKey-')
		AlphaKey = float(st.session_state['-AlphaKey-'])

	####################################################################################

	st.markdown("")

	submitted = st.form_submit_button('Analyze')

	####################################################################################

	if uploaded_file is None:
		st.stop()

	raw_image_from_pillow = Image.open(uploaded_file)

	raw_image = np.array(raw_image_from_pillow)

	if len(raw_image.shape) != 2:
		raise ValueError("Invalid image format")

	####################################################################################

	if submitted:

		try:
			
			filtered_image = skimage.filters.gaussian(raw_image, sigma = FilterKey, mode = 'nearest', preserve_range = True)

			image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

			Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

			Image_Coherance = make_coherence(filtered_image, EigenValues, ThresholdValueKey)

			Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)

			vx, vy = make_vxvy(filtered_image, EigenVectors, ThresholdValueKey)

		except:

			raise Exception('Something went wrong in the analysis')

		####################################################################################

		FIGSIZE = (6, 6)

		left_column3, right_column3  = st.columns(2)

		with left_column3:
	
			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
			im = plt.imshow(raw_image, vmin = 0, vmax = 255, cmap = 'viridis')

			plt.title('Uploaded Image', pad = PAD, fontsize = FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			aspect = 20
			pad_fraction = 0.5

			ax = plt.gca()
			divider = make_axes_locatable(ax)
			width = axes_size.AxesY(ax, aspect=1./aspect)
			pad = axes_size.Fraction(pad_fraction, width)
			cax = divider.append_axes("right", size=width, pad=pad)
			cbar = plt.colorbar(im, cax=cax)
			cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

			st.pyplot(fig)

		#########

		with right_column3:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
			im = plt.imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'viridis')

			plt.title('Filtered Image', pad = PAD, fontsize = FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			aspect = 20
			pad_fraction = 0.5

			ax = plt.gca()
			divider = make_axes_locatable(ax)
			width = axes_size.AxesY(ax, aspect=1./aspect)
			pad = axes_size.Fraction(pad_fraction, width)
			cax = divider.append_axes("right", size=width, pad=pad)
			cbar = plt.colorbar(im, cax=cax)
			cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

			st.pyplot(fig)

		#########

		left_column4, right_column4 = st.columns(2)

		with left_column4:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
			im = plt.imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'RdYlBu_r')

			plt.title('Coherence', pad = PAD, fontsize = FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			aspect = 20
			pad_fraction = 0.5

			ax = plt.gca()
			divider = make_axes_locatable(ax)
			width = axes_size.AxesY(ax, aspect=1./aspect)
			pad = axes_size.Fraction(pad_fraction, width)
			cax = divider.append_axes("right", size=width, pad=pad)
			cbar = plt.colorbar(im, cax=cax)
			cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

			st.pyplot(fig)

		#########

		with right_column4:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
			im = plt.imshow(Image_Orientation/180, vmin = 0, vmax = 1, cmap = 'hsv')

			plt.title('Orientation', pad = PAD, fontsize = FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			aspect = 20
			pad_fraction = 0.5

			ax = plt.gca()
			divider = make_axes_locatable(ax)
			width = axes_size.AxesY(ax, aspect=1./aspect)
			pad = axes_size.Fraction(pad_fraction, width)
			cax = divider.append_axes("right", size=width, pad=pad)
			cbar = fig.colorbar(im, cax = cax, ticks = np.linspace(0, 1, 5))
			cbar.ax.set_yticklabels([r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', r'$135^{\circ}$', r'$180^{\circ}$'])
			ticklabs = cbar.ax.get_yticklabels()
			cbar.ax.set_yticklabels(ticklabs, fontsize = FONTSIZE_TITLE)

			st.pyplot(fig)

		#########

		left_column5, middle_column5, right_column5  = st.columns(3)

		with left_column5:

			st.markdown('')

		with middle_column5:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

			plt.imshow(raw_image, cmap = 'Oranges', alpha = AlphaKey)

			xmesh, ymesh = np.meshgrid(np.arange(raw_image.shape[0]), np.arange(raw_image.shape[1]), indexing = 'ij')

			plt.quiver(ymesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], xmesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vy[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], vx[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
			scale = ScaleKey, headlength = 0, headaxislength = 0, 
			pivot = 'middle', color = 'k', angles = 'xy')

			plt.title('Local Orientation', pad = PAD, fontsize = 1.3*FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			st.pyplot(fig)

		with right_column5:

			st.markdown('')

		########################################################################

		st.stop()
