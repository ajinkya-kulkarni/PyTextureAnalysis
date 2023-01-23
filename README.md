## Overview

PyTextureAnalysis is a Python package (inspired from http://bigwww.epfl.ch/demo/orientation/) that contains tools to analyze the texture of images. This code contains functions to calculate the local orientation of fibres in an image, as well as the degree of coherence, which is a measure of how bundled and organized the fibres are. The following functions are included:

- `make_coherence`: calculates the degree of coherence of fibres in an image based on the eigenvalues and structure tensor of the image.
- `make_image_gradients`: calculates the gradients of an image in the x and y directions using Gaussian filters.
- `make_orientation`: calculates the local orientation of fibres in an image based on the Jxx, Jxy, and Jyy components of the structure tensor.
- `make_vxvy`: calculates the x and y components of the eigenvectors of the structure tensor.
- `make_structure_tensor_2d`: calculates the 2D structure tensor of an image using the image gradients in the x and y directions and a local standard deviation.

## Demo

Have a look at the `Example.ipynb` file to get an idea how to use the package to extract and visualize local fibre orientation and organization.

## Dependencies

The following dependencies must be installed:

- Python
- numpy
- scipy
- opencv-python

You can install these dependencies by first installing `Python` and then running the command: `pip install -r requirements.txt` in the terminal.
