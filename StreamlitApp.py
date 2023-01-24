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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 12})

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

st.set_page_config(page_title = 'PyTextureAnalysis', page_icon = image_bytes, layout = "centered", initial_sidebar_state = "expanded", menu_items = {'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de', 'About': 'This is a application for demonstrating the PyTextureAnalysis package. Developed, tested and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen'})

# Title of the web app

st.title(':blue[Application for demonstrating the PyTextureAnalysis package]')

st.markdown("")

#######################################################################################################

def main():

	uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff"], accept_multiple_files = False, label_visibility = 'visible')

	st.markdown("")
	st.markdown("""---""")
	st.markdown("")

	if uploaded_file is not None:

		raw_image = cv.imread(uploaded_file.name, cv.IMREAD_GRAYSCALE)

		st.subheader('Uploaded Image')
		st.image(raw_image, use_column_width = True)

		st.markdown("")

		st.slider('Filter', min_value = 0.0, max_value = 10.0, value = 1.0, step = 0.5, format = '%0.1f', label_visibility = "visible", key = '-FilterKey-')
		FilterKey = float(st.session_state['-FilterKey-'])

		st.slider('Local Sigma', min_value = 0, max_value = 20, value = 5, step = 1, format = '%d', label_visibility = "visible", key = '-LocalSigmaKey-')
		LocalSigmaKey = int(st.session_state['-LocalSigmaKey-'])

		st.slider('Threshold Value', min_value = 0, max_value = 200, value = 10, step = 5, format = '%d', label_visibility = "visible", key = '-ThresholdValueKey-')
		ThresholdValueKey = int(st.session_state['-ThresholdValueKey-'])

		filtered_image = skimage.filters.gaussian(raw_image, sigma = FilterKey, mode = 'nearest', preserve_range = True)
		plt.clf()
		
		im = plt.imshow(filtered_image, vmin = 0, vmax = 255, cmap = 'viridis')
		plt.axis('off')

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "5%", pad="3%")
		plt.colorbar(im, cax=cax)
		plt.tight_layout()
		plt.savefig('Filtered_image.png', dpi = 400, bbox_inches='tight')

		st.subheader('Filtered Image')
		st.image('Filtered_image.png', clamp = True, use_column_width = True)

		st.markdown("")

		image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

		Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x, image_gradient_y, LocalSigmaKey)

		Image_Coherance = make_coherence(filtered_image, EigenValues, ThresholdValueKey)

		Image_Orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, ThresholdValueKey)

		vx, vy = make_vx_vy(filtered_image, EigenVectors, ThresholdValueKey)

		########################################################################

		plt.clf()
		im = plt.imshow(Image_Coherance, vmin = 0, vmax = 1, cmap = 'RdYlBu_r')
		plt.axis('off')

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "5%", pad="3%")
		plt.colorbar(im, cax=cax)
		plt.tight_layout()
		plt.savefig('Coherance.png', dpi = 400, bbox_inches='tight')

		st.subheader('Coherance')
		st.image('Coherance.png', use_column_width=True)

		st.markdown("")

		########################################################################

		plt.clf()
		im = plt.imshow(Image_Orientation, vmin = 0, vmax = 180, cmap = 'hsv')
		plt.axis('off')

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "5%", pad="3%")
		plt.colorbar(im, cax=cax)
		plt.tight_layout()
		plt.savefig('Orientation.png', dpi = 400, bbox_inches='tight')

		st.subheader('Orientation')
		st.image('Orientation.png', use_column_width=True)

		st.markdown("")

		########################################################################

		st.slider('Spacing', min_value = 5, max_value = 50, value = 20, step = 5, format = '%d', label_visibility = "visible", key = '-SpacingKey-')
		SpacingKey = int(st.session_state['-SpacingKey-'])

		st.slider('Scale', min_value = 10, max_value = 100, value = 60, step = 5, format = '%d', label_visibility = "visible", key = '-ScaleKey-')
		ScaleKey = int(st.session_state['-ScaleKey-'])

		plt.clf()
		plt.imshow(raw_image, cmap = 'Oranges', alpha = 0.7)
		xmesh, ymesh = np.meshgrid(np.arange(raw_image.shape[0]), np.arange(raw_image.shape[1]), indexing = 'ij')
		plt.quiver(ymesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], 
					xmesh[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
					vy[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey], 
					vx[SpacingKey//2::SpacingKey, SpacingKey//2::SpacingKey],
					scale = ScaleKey, headlength = 0, headaxislength = 0, 
					pivot = 'middle', color = 'k', angles = 'xy')

		plt.axis('off')
		plt.tight_layout()
		plt.savefig('OrientationVectorField.png', dpi = 400, bbox_inches='tight')

		st.subheader('Local Orientation')
		st.image('OrientationVectorField.png', use_column_width=True)
		
		########################################################################

		st.stop()

if __name__== "__main__":
	main()
