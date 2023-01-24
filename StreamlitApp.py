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
from scipy import ndimage
import scipy.ndimage
import skimage as skimage

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

# Title of the web app

st.title(':blue[Application for demonstrating the PyTextureAnalysis package]')

st.markdown("")

#######################################################################################################

def main():

	uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff"], accept_multiple_files = False, label_visibility = 'visible')

	if uploaded_file is not None:

		raw_image = cv.imread(uploaded_file.name, cv.IMREAD_GRAYSCALE)

		st.image(raw_image, caption='Original Image.', use_column_width=True, clamp=True)

		image_filter_sigma = 1
		local_sigma = 8
		threshold_value = max(int(0.5 * np.median(raw_image)), 2)

		filtered_image = skimage.filters.gaussian(raw_image, sigma = image_filter_sigma, mode = 'nearest', preserve_range = True)

		image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

		Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, local_sigma)

		Image_Coherance = make_coherence(filtered_image, EigenValues, threshold_value)

		Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, threshold_value)

		vx, vy = make_vx_vy(filtered_image, EigenVectors, threshold_value)

		left_column1, right_column1  = st.columns(2)

		with left_column1:

			st.image(Image_Coherance, caption='Coherance', use_column_width=True)

		with right_column1:

			st.image(Image_Orientation, caption='Orientation', use_column_width=True)

if __name__== "__main__":
	main()
