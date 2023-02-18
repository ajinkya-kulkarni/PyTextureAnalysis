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

from modules import *

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

FIGSIZE = (5, 5)
PAD = 10
FONTSIZE_TITLE = 15
DPI = 500

aspect = 20
pad_fraction = 0.5

factor = 1.2
markersize = 12
linewidth = 3

each_chunk_size = 20
Trigger_for_chunks = 20

########################################################################################

with st.form(key = 'form1', clear_on_submit = False):
	
	st.markdown(':blue[Upload a 2D grayscale image to be analyzed. Works best with images smaller than 600x600 pixels.]')

	uploaded_file = st.file_uploader("Upload a 2D grayscale image to be analyzed. Works best with images smaller than 600x600 pixels.", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	st.markdown("""---""")
	
	left_column1, middle_column1, right_column1  = st.columns(3)

	with left_column1:
		st.slider('Gaussian image filter sigma [pixels]', min_value = 1, max_value = 10, value = 1, step = 1, format = '%d', label_visibility = "visible", key = '-FilterKey-')
		FilterKey = int(st.session_state['-FilterKey-'])

	with middle_column1:
		st.slider('Gaussian local window [pixels]', min_value = 1, max_value = 100, value = 10, step = 1, format = '%d', label_visibility = "visible", key = '-LocalSigmaKey-')
		LocalSigmaKey = int(st.session_state['-LocalSigmaKey-'])

	with right_column1:
		st.slider('Window size for evaluating local density [pixels]', min_value = 1, max_value = 100, value = 20, step = 1, format = '%d', label_visibility = "visible", key = '-LocalDensityKey-')
		LocalDensityKey = int(st.session_state['-LocalDensityKey-'])

	####################################################################################

	left_column2, middle_column2, right_column2  = st.columns(3)

	with left_column2:

		st.slider('Threshold value for pixel evaluation [pixels]' , min_value = 5, max_value = 200, value = 20, step = 5, format = '%d', label_visibility = "visible", key = '-ThresholdValueKey-')
		ThresholdValueKey = int(st.session_state['-ThresholdValueKey-'])

	with middle_column2:
		st.slider('Spacing between the orientation vectors', min_value = 5, max_value = 50, value = 20, step = 5, format = '%d', label_visibility = "visible", key = '-SpacingKey-')
		SpacingKey = int(st.session_state['-SpacingKey-'])

	with right_column2:
		st.slider('Scaling for the orientation vectors', min_value = 10, max_value = 100, value = 40, step = 5, format = '%d', label_visibility = "visible", key = '-ScaleKey-')
		ScaleKey = int(st.session_state['-ScaleKey-'])
		
	####################################################################################

	st.markdown("")

	submitted = st.form_submit_button('Analyze')

	st.markdown("")

	####################################################################################

	if uploaded_file is None:
		st.stop()
		
	####################################################################################

	if submitted:

		ProgressBarText = st.empty()
		ProgressBarText.caption("Analyzing...")
		ProgressBar = st.progress(0)
		ProgressBarTime = 0.1

		try:

			# Read the image
			raw_image = convert_to_8bit_grayscale(uploaded_file)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(1/6))

			# Filter the image
			filtered_image = skimage.filters.gaussian(raw_image, sigma = FilterKey, mode = 'nearest', preserve_range = True)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(2/6))

			###########################

			# Calculate local density by binarizing the image first (using simple mean thresholding), then convoluting it with a nxn kernel of ones. 
			# Currently the kernel size is equal to the local window used for calculating coherence and orientation.
			# Please refer to: https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-14-25718&id=477526 for more information.

			# Binarize the image
			binarized_image = binarize_image(filtered_image)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(3/6))

			# Define the kernel and it's size
			local_kernel_size = LocalDensityKey
			if (local_kernel_size % 2 == 0):
				local_kernel_size = local_kernel_size + 1
			if (local_kernel_size < 3):
				local_kernel_size = 3

			local_kernel = np.ones((local_kernel_size, local_kernel_size), dtype = np.float32) / (local_kernel_size * local_kernel_size)

			Local_Density = convolve(binarized_image, local_kernel)

			Local_Density = np.divide(Local_Density, Local_Density.max(), out=np.full(Local_Density.shape, np.nan), where=Local_Density.max() != 0)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(4/6))

			###########################

			# Trigger for evaluating the images as chunks. Currently activated only for each_chunk_size > 20, or if the 2D input image is bigger than 600x600 pixels.

			if ((each_chunk_size >= Trigger_for_chunks) or (min(raw_image.shape[0], raw_image.shape[1]) > 600)):

				pass

				# padded_raw_image = generate_padded_image(raw_image, each_chunk_size)
				# chunks = split_into_chunks(padded_raw_image, each_chunk_size)

				# Image_Coherance_list = []
				# Image_Orientation_list = []
				# vx_list = []
				# vy_list = []

				# for i in range(len(chunks)):

				# 	current_chunk = chunks[i]

				# 	filtered_chunk = skimage.filters.gaussian(current_chunk,
				# 											sigma = FilterKey,
				# 											mode = 'nearest',
				# 											preserve_range = True)

				# 	# Calculate image gradients in X and Y directions
				# 	image_gradient_x, image_gradient_y = make_image_gradients(filtered_chunk)

				# 	# Calculate the structure tensor and solve for EigenValues, EigenVectors

				# 	Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

				# 	Image_Coherance = make_coherence(filtered_chunk, EigenValues, Structure_Tensor, ThresholdValueKey)

				# 	Image_Orientation = make_orientation(filtered_chunk, Jxx, Jxy, Jyy, ThresholdValueKey)
				# 	vx, vy = make_vxvy(filtered_chunk, EigenVectors, ThresholdValueKey)

				# 	Image_Coherance_list.append(Image_Coherance)
				# 	Image_Orientation_list.append(Image_Orientation)
				# 	vx_list.append(vx)
				# 	vy_list.append(vy)

				# ###########################

				# Image_Orientation = stitch_back_chunks(Image_Orientation_list, padded_raw_image, raw_image, each_chunk_size)

				# Image_Coherance = stitch_back_chunks(Image_Coherance_list, padded_raw_image, raw_image, each_chunk_size)

				# vx = stitch_back_chunks(vx_list, padded_raw_image, raw_image, each_chunk_size)
				# vy = stitch_back_chunks(vy_list, padded_raw_image, raw_image, each_chunk_size)

				# time.sleep(ProgressBarTime)
				# ProgressBar.progress(float(5/6))

			# else:

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

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(5/6))

			###########################

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(6/6))

			time.sleep(ProgressBarTime)

			ProgressBarText.empty()
			ProgressBar.empty()

		except:

			raise Exception('Analysis unsuccessful')

		####################################################################################

		left_column3, middle_column3, right_column3  = st.columns(3)

		with left_column3:
	
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

			st.pyplot(fig)

		with middle_column3:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
# 			im = plt.imshow(binarized_image, vmin = 0, vmax = 1, cmap = 'binary')
			im = plt.imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'binary')

			plt.title('Binarized Image', pad = PAD, fontsize = FONTSIZE_TITLE)
			plt.xticks([])
			plt.yticks([])

			ax = plt.gca()
			divider = make_axes_locatable(ax)
			width = axes_size.AxesY(ax, aspect=1./aspect)
			pad = axes_size.Fraction(pad_fraction, width)
			cax = divider.append_axes("right", size=width, pad=pad)
			cbar = plt.colorbar(im, cax=cax)
			cbar.ax.tick_params(labelsize = FONTSIZE_TITLE)

			st.pyplot(fig)

		with right_column3:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)
			im = plt.imshow(Local_Density, vmin = 0, vmax = 1, cmap = 'Spectral_r')

			plt.title('Local Density', pad = PAD, fontsize = FONTSIZE_TITLE)
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

			st.pyplot(fig)

		#########

		left_column4, middle_column4, right_column4 = st.columns(3)

		with left_column4:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

			#im = plt.imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'Spectral_r')
			im = plt.imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.Spectral_r(Image_Coherance), vmin = 0, vmax = 1, cmap = 'Spectral_r')
			
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

			st.pyplot(fig)

		with middle_column4:

			fig = plt.figure(figsize = FIGSIZE, constrained_layout = True, dpi = DPI)

			#im = plt.imshow(Image_Orientation/180, vmin = 0, vmax = 1, cmap = 'hsv')
			im = plt.imshow(plt.cm.binary_r(binarized_image/binarized_image.max()) * plt.cm.hsv(Image_Orientation/180), vmin = 0, vmax = 1, cmap = 'hsv')

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

			st.pyplot(fig)

		with right_column4:

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

			st.pyplot(fig)

		########################################################################

		st.stop()
