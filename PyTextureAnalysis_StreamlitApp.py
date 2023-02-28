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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import time
from io import BytesIO

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

########################################################################################

from modules import *
from parameters import *

########################################################################################

with open("logo.jpg", "rb") as f:
	image_data = f.read()

image_bytes = BytesIO(image_data)

st.set_page_config(page_title = 'PyTextureAnalysis', page_icon = image_bytes, layout = "wide", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PyTextureAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

# Title of the web app

st.title(':blue[Texture analysis using PyTextureAnalysis]')

st.caption('For more information, have a look at this [screenshot](https://github.com/ajinkya-kulkarni/PyTextureAnalysis/blob/main/StreamlitApp.jpg). Source code available [here](https://github.com/ajinkya-kulkarni/PyTextureAnalysis).', unsafe_allow_html = False)

st.markdown("")

########################################################################################

with st.form(key = 'form1', clear_on_submit = False):
	
	st.markdown(':blue[Upload a 2D grayscale image to be analyzed. Works best with images smaller than 600x600 pixels.]')

	uploaded_file = st.file_uploader("Upload a 2D grayscale image to be analyzed. Works best with images smaller than 600x600 pixels.", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

	st.markdown("""---""")
	
	left_column1, middle_column1, right_column1  = st.columns(3)

	with left_column1:
		st.slider('Gaussian image filter sigma [pixels]', min_value = 1, max_value = 10, value = 2, step = 1, format = '%d', label_visibility = "visible", key = '-FilterKey-')
		FilterKey = int(st.session_state['-FilterKey-'])

	with middle_column1:
		st.slider('Gaussian local window [pixels]', min_value = 1, max_value = 100, value = 10, step = 1, format = '%d', label_visibility = "visible", key = '-LocalSigmaKey-')
		LocalSigmaKey = int(st.session_state['-LocalSigmaKey-'])

	with right_column1:
		st.slider('Window size for evaluating local density [pixels]', min_value = 5, max_value = 100, value = 20, step = 5, format = '%d', label_visibility = "visible", key = '-LocalDensityKey-')
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
			ProgressBar.progress(float(1/11))

			###########################

			# Filter the image
			filtered_image = make_filtered_image(raw_image, FilterKey)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(2/11))

			###########################

			# Calculate local density by binarizing the image first (using simple mean thresholding), then convoluting it with a nxn kernel of ones. 
			# Currently the kernel size is equal to the local window used for calculating coherence and orientation.
			# Please refer to: https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-14-25718&id=477526 for more information.

			# Binarize the image

			BinarizationKey = 20

			binarized_image = binarize_image(filtered_image, radius = BinarizationKey)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(3/11))

			###########################

			# Calculate the fibrotic_percentage area of the non-zero pixels compared to the image size
			fibrotic_percentage = percentage_area(binarized_image)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(4/11))

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

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(5/11))

			###########################

			# Calculate image gradients in X and Y directions
			image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(6/11))

			###########################

			# Calculate the structure tensor and solve for EigenValues, EigenVectors
			Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(7/11))

			###########################

			# Calculate Coherence

			Image_Coherance = make_coherence(filtered_image, EigenValues, Structure_Tensor, ThresholdValueKey)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(8/11))

			###########################

			# Calculate Orientation
			Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)
			vx, vy = make_vxvy(filtered_image, EigenVectors, ThresholdValueKey)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(9/11))

		####################################################################################

			fig = make_mosiac_plot(raw_image, binarized_image, filtered_image, Local_Density, Image_Coherance, Image_Orientation, vx, vy, filename, LocalSigmaKey, fibrotic_percentage, SpacingKey, ScaleKey, FIGSIZE, 2*DPI, PAD, FONTSIZE_TITLE, pad_fraction, aspect)

			st.pyplot(fig)

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(10/11))

			# Perform statistical analysis

			results_array = perform_statistical_analysis(filename, LocalSigmaKey, Image_Orientation, Image_Coherance)

			dataframe = load_pandas_dataframe(results_array)

			st.markdown("")

			st.markdown("Detailed Report")

			st.dataframe(dataframe, use_container_width=True)

			####################################################################################

			time.sleep(ProgressBarTime)
			ProgressBar.progress(float(11/11))

			ProgressBarText.empty()
			ProgressBar.empty()
		except:

			raise Exception('Analysis unsuccessful')
			
		########################################################################

		st.stop()
