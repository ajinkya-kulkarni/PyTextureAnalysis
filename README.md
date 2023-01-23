# Package Name
PyTextureAnalysis

# Summary
PyTextureAnalysis is a package that contains tools to analyze the texture of images, including functions for calculating local orientation, degree of coherence, and structure tensor of an image.

## Overview

This code contains functions to calculate the local orientation of fibres in an image, as well as the degree of coherence, which is a measure of how bundled and organized the fibres are. The following functions are included:

- `make_coherence`: calculates the degree of coherence of fibres in an image based on the eigenvalues and structure tensor of the image.
- `make_image_gradients`: calculates the gradients of an image in the x and y directions using Gaussian filters.
- `make_orientation`: calculates the local orientation of fibres in an image based on the Jxx, Jxy, and Jyy components of the structure tensor.
- `make_vxvy`: calculates the x and y components of the eigenvectors of the structure tensor.
- `make_structure_tensor_2d`: calculates the 2D structure tensor of an image using the image gradients in the x and y directions and a local standard deviation.

## Dependencies

The following dependencies must be installed:

- numpy
- scipy
- opencv-python

You can install these dependencies by running the command:
pip install -r requirements.txt
